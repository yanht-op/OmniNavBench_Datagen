"""Bake a Z-up scene with OmniNavBench settings and NVIDIA Navigation APIs.

Run with Isaac Sim 5.0 Python from the repository root. Outputs a report and,
when supported, a visualization-only USD triangle mesh, NOT native NavMesh data.
"""
import argparse
import faulthandler
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

# Direct script execution must be able to import the repository packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def validate_bounds(values):
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise ValueError('Bounds must contain six finite meter coordinates')
    if any(values[i + 3] <= values[i] for i in range(3)):
        raise ValueError('Bounds require MAX > MIN on every axis')
    return [values[i + 3] - values[i] for i in range(3)]


def wait_for_bake(interface, update, timeout, state):
    """Keep Kit updating while the native asynchronous bake runs."""
    started = interface.start_navmesh_baking()
    if started is False:
        raise RuntimeError('Navigation extension rejected the bake request')
    begin = last_log = time.monotonic()
    while True:
        update()
        if state.get('status') is False:
            raise RuntimeError('Navigation reported bake failure/cancellation')
        busy = interface.is_navmesh_baking()
        if not busy:
            mesh = interface.get_navmesh()
            if mesh is not None:
                return mesh
            raise RuntimeError('Bake finished without a NavMesh')
        now = time.monotonic()
        if now - begin > timeout:
            interface.cancel_navmesh_baking()
            raise TimeoutError('Bake timeout; cancellation requested')
        if now - last_log >= 15:
            print(f'Baking: elapsed={now-begin:.0f}s, progress={state.get("progress", "unavailable")}', flush=True)
            last_log = now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-usd', type=Path, required=True)
    parser.add_argument('--source-root', default='/Root', help='Root referenced from source USD')
    parser.add_argument('--geometry-root', default='/Root/Meshes', help='Source subtree used for automatic bounds')
    parser.add_argument('--bounds', type=float, nargs=6, metavar=('XMIN','YMIN','ZMIN','XMAX','YMAX','ZMAX'), help='Explicit bake include bounds in WORLD METERS; replaces automatic bounds')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--agent-radius', type=float, default=.35)
    parser.add_argument('--max-step-height', type=float, default=.05)
    parser.add_argument('--cell-size', type=float, default=.1)
    parser.add_argument('--cell-height', type=float, default=.1)
    parser.add_argument('--max-span', type=float, default=100, help='Reject any bake axis exceeding this meter length')
    parser.add_argument('--timeout', type=float, default=900, help='Bake timeout; cannot interrupt a blocked native update')
    parser.add_argument('--mdl-path', type=Path, action='append', default=[])
    parser.add_argument('--inspect-only', action='store_true', help='Report geometry bounds and exit before baking')
    args = parser.parse_args()
    for value in (args.agent_radius, args.cell_size, args.cell_height, args.max_span, args.timeout):
        if not math.isfinite(value) or value <= 0:
            parser.error('Radius, cell sizes, max-span and timeout must be finite and positive')
    if not math.isfinite(args.max_step_height) or args.max_step_height < 0:
        parser.error('Step height must be finite and nonnegative')
    if args.bounds:
        validate_bounds(args.bounds)
    scene = args.scene_usd.expanduser().resolve()
    if not scene.is_file():
        parser.error(f'Scene missing: {scene}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [x.expanduser().resolve() for x in args.mdl_path]
    if (scene.parent/'Materials'/'KooPbr.mdl').is_file():
        paths.append(scene.parent/'Materials')
    if any(not x.is_dir() for x in paths):
        parser.error('MDL path must be a directory')
    if paths:
        os.environ['MDL_USER_PATH'] = os.pathsep.join([str(x) for x in paths] + [os.environ.get('MDL_USER_PATH') or str(Path.home()/'Documents'/'mdl')])
    from isaacsim import SimulationApp
    from OmniNav.local_paths import resolve_runtime_kit_path
    faulthandler.dump_traceback_later(180, repeat=True)
    print('[1/5] Starting project runtime', flush=True)
    app = SimulationApp({'headless': True, 'multi_gpu': False}, experience=str(resolve_runtime_kit_path()))
    temporary = tempfile.TemporaryDirectory(prefix='omninav-bake-')
    report = {'scene_usd': str(scene), 'bake_success': False}
    report_path = args.output_dir/'bake_report.json'
    try:
        import omni.usd
        import omni.kit.commands
        import omni.anim.navigation.core as nav
        from pxr import Usd, UsdGeom, Gf, Sdf
        from OmniNavExt.envset.core.scene_manager import SceneManager
        print('[2/5] Composing meter-scale geometry wrapper', flush=True)
        source = Usd.Stage.Open(str(scene))
        units = float(UsdGeom.GetStageMetersPerUnit(source))
        if not math.isfinite(units) or units <= 0 or UsdGeom.GetStageUpAxis(source) != UsdGeom.Tokens.z:
            raise ValueError('Source must have positive metersPerUnit and Z-up')
        source_root = source.GetPrimAtPath(args.source_root)
        if not source_root:
            raise ValueError(f'Source root missing: {args.source_root}')
        for prim in Usd.PrimRange(source_root, Usd.TraverseInstanceProxies()):
            xf = UsdGeom.Xformable(prim)
            if xf and xf.GetResetXformStack():
                raise ValueError(f'Parent unit scaling is unsafe with resetXformStack: {prim.GetPath()}')
        wrapper_path = Path(temporary.name)/'scene.usda'
        wrapper = Usd.Stage.CreateNew(str(wrapper_path))
        UsdGeom.SetStageMetersPerUnit(wrapper, 1.)
        UsdGeom.SetStageUpAxis(wrapper, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(wrapper, '/World')
        wrapper.SetDefaultPrim(world.GetPrim())
        scale = UsdGeom.Xform.Define(wrapper, '/World/MetricScale')
        scale.AddScaleOp().Set(Gf.Vec3f(units, units, units))
        wrapper.DefinePrim('/World/MetricScale/Scene').GetReferences().AddReference(str(scene), source_root.GetPath())
        wrapper.GetRootLayer().Save()
        context = omni.usd.get_context()
        if not context.open_stage(str(wrapper_path)):
            raise RuntimeError('Cannot open wrapper stage')
        deadline = time.monotonic() + args.timeout
        while True:
            app.update()
            if context.get_stage() and context.get_stage_loading_status()[2] == 0:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Stage loading timeout')
        stage = context.get_stage()
        source_prefix = Sdf.Path(args.source_root)
        geometry_path = Sdf.Path(args.geometry_root)
        if not geometry_path.HasPrefix(source_prefix):
            raise ValueError('--geometry-root must be inside --source-root')
        mapped = geometry_path.ReplacePrefix(source_prefix, Sdf.Path('/World/MetricScale/Scene'))
        root = stage.GetPrimAtPath(mapped)
        if not root:
            raise ValueError(f'Geometry root missing: {args.geometry_root}')
        # Ignore camera/light bounds and existing NavMesh volumes.
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render','proxy'])
        box = Gf.Range3d()
        largest = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not UsdGeom.Boundable(prim) or prim.GetTypeName().startswith('NavMesh'):
                continue
            b = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if b.IsEmpty():
                continue
            box.UnionWith(b)
            lo, hi = b.GetMin(), b.GetMax()
            largest.append({'path': str(prim.GetPath()), 'min_m': list(lo), 'max_m': list(hi), 'span_m': list(hi-lo)})
        if box.IsEmpty():
            raise ValueError('No geometry bounds found')
        largest.sort(key=lambda row: max(row['span_m']), reverse=True)
        lo, hi = box.GetMin(), box.GetMax()
        bounds = args.bounds or [lo[0]-.2,lo[1]-.2,lo[2]-.2,hi[0]+.2,hi[1]+.2,hi[2]+.2]
        size = validate_bounds(bounds)
        report.update(source_meters_per_unit=units, bounds_m=bounds, largest_geometry_bounds=largest[:20], agent_radius_m=args.agent_radius, max_step_height_m=args.max_step_height, cell_size_m=args.cell_size, cell_height_m=args.cell_height)
        print(f'[3/5] Bake bounds (meters): {bounds}; size={size}', flush=True)
        for row in largest[:5]:
            print(f'Large geometry: {row}', flush=True)
        if args.inspect_only:
            report['inspection_only'] = True
            return
        if max(size) > args.max_span:
            raise ValueError(f'Bake span exceeds {args.max_span} m. Inspect bake_report.json and set correct --bounds; do not blindly enlarge the limit.')
        # Replace old include ranges in memory, preserve exclusion volumes.
        for prim in list(stage.Traverse()):
            if prim.GetTypeName() in ('NavMeshVolume','NavMeshIncludeVolume'):
                prim.SetActive(False)
        parent = '/World/BakeVolume'
        stage.DefinePrim(parent, 'Xform')
        before_paths = {str(x.GetPath()) for x in stage.Traverse()}
        command_result = omni.kit.commands.execute('CreateNavMeshVolumeCommand', parent_prim_path=Sdf.Path(parent), volume_type=0, layer=stage.GetRootLayer())
        # Some Kit versions create the volume outside parent_prim_path. The
        # project's navmesh_utils handles this too; discover by stage delta.
        added = [x for x in stage.Traverse() if str(x.GetPath()) not in before_paths]
        report['volume_command_result'] = repr(command_result)
        report['created_prims'] = [{'path': str(x.GetPath()), 'type': x.GetTypeName(), 'schemas': list(x.GetAppliedSchemas())} for x in added]
        print(f'Volume creation: {report["volume_command_result"]}; created={report["created_prims"]}', flush=True)
        volumes = [x for x in added if x.GetTypeName() in ('NavMeshVolume','NavMeshIncludeVolume')]
        if len(volumes) != 1:
            raise RuntimeError(f'Expected one new include volume anywhere in stage, found {len(volumes)}; inspect created_prims and volume_command_result in report')
        report['include_volume_path'] = str(volumes[0].GetPath())
        xf = UsdGeom.Xformable(volumes[0])
        xf.ClearXformOpOrder()
        # Bounds below are world meters even if Kit chooses a scaled parent.
        xf.SetResetXformStack(True)
        xf.AddTranslateOp().Set(Gf.Vec3d(*[(bounds[i]+bounds[i+3])/2 for i in range(3)]))
        xf.AddScaleOp().Set(Gf.Vec3d(*size))
        SceneManager._set_navmesh_config(args.agent_radius, args.max_step_height)
        for key, value in [('cellSize',args.cell_size),('cellHeight',args.cell_height)]:
            omni.kit.commands.execute('ChangeSetting', path='/exts/omni.anim.navigation.core/navMesh/config/'+key, value=value)
        for _ in range(5):
            app.update()
        interface = nav.acquire_interface()
        state, subscriptions = {}, []
        # latest docs use Events 2.0; installed 5.0 may expose older APIs.
        if hasattr(nav, 'GLOBAL_EVENT_NAVMESH_UPDATED'):
            import carb.eventdispatcher
            dispatcher = carb.eventdispatcher.get_eventdispatcher()
            def completed(event):
                state['status'] = event.payload.get('status')
            def progress(event):
                state['progress'] = event.payload.get('progress')
            subscriptions.append(dispatcher.observe_event(observer_name='bake_complete', event_name=nav.GLOBAL_EVENT_NAVMESH_UPDATED, on_event=completed))
            if hasattr(nav, 'GLOBAL_EVENT_NAVMESH_UPDATING'):
                subscriptions.append(dispatcher.observe_event(observer_name='bake_progress', event_name=nav.GLOBAL_EVENT_NAVMESH_UPDATING, on_event=progress))
        print('[4/5] Starting native NavMesh bake', flush=True)
        if all(hasattr(interface, name) for name in ('start_navmesh_baking','is_navmesh_baking','cancel_navmesh_baking')):
            mesh = wait_for_bake(interface, app.update, args.timeout, state)
        else:
            print('Installed API requires blocking bake; Python timeout cannot interrupt it', flush=True)
            interface.start_navmesh_baking_and_wait()
            mesh = interface.get_navmesh()
            if mesh is None or state.get('status') is False:
                raise RuntimeError('Bake failed: no successful NavMesh')
        report['bake_success'] = True
        if hasattr(interface, 'get_cache_dir'):
            report['native_cache_directory'] = str(interface.get_cache_dir())
        print('[5/5] Bake succeeded', flush=True)
        if hasattr(mesh, 'get_draw_triangles') and hasattr(mesh, 'get_area_count'):
            visual_path = args.output_dir/'navmesh_visualization.usda'
            visual = Usd.Stage.CreateNew(str(visual_path))
            UsdGeom.SetStageMetersPerUnit(visual, 1.)
            UsdGeom.SetStageUpAxis(visual, UsdGeom.Tokens.z)
            visual.SetDefaultPrim(UsdGeom.Xform.Define(visual, '/NavMeshVisualization').GetPrim())
            total = 0
            for area in range(mesh.get_area_count()):
                vertices = mesh.get_draw_triangles(area)
                if not vertices:
                    continue
                if len(vertices) % 3:
                    raise RuntimeError('Unexpected triangle vertex count')
                points = [Gf.Vec3f(v.x,v.y,v.z) if hasattr(v,'x') else Gf.Vec3f(*v) for v in vertices]
                output_mesh = UsdGeom.Mesh.Define(visual, f'/NavMeshVisualization/Area_{area}')
                output_mesh.CreatePointsAttr(points)
                output_mesh.CreateFaceVertexCountsAttr([3]*(len(points)//3))
                output_mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
                output_mesh.CreateSubdivisionSchemeAttr('none')
                output_mesh.CreateDoubleSidedAttr(True)
                output_mesh.CreateDisplayColorAttr([Gf.Vec3f(.1,.65,.8)])
                total += len(points)//3
            visual.GetRootLayer().Save()
            report.update(visualization_triangle_count=total, visualization_usd=str(visual_path.resolve()), visualization_note='Debug drawing triangles only; not a serialized queryable NavMesh')
            print(f'Visualization triangles: {total}; saved {visual_path.resolve()}', flush=True)
        else:
            report['visualization_note'] = 'Installed API has no get_draw_triangles/get_area_count; bake succeeded, visualization unavailable'
    except BaseException as exc:
        report['error'] = str(exc)
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Report: {report_path.resolve()}', flush=True)
        faulthandler.cancel_dump_traceback_later()
        app.close()
        temporary.cleanup()


if __name__ == '__main__':
    main()
