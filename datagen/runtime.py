"""Isaac lifecycle for collection, reusing the benchmark's replay setup helpers."""
from __future__ import annotations

import json
import faulthandler
from datetime import datetime, timezone
from time import monotonic
import traceback


def collect(args, config, source_envset):
    """Record stacks from inside this process; no ptrace/sudo is required."""
    stack_path = args.output / "python_stacks.log"
    with stack_path.open("x", encoding="utf-8") as stacks:
        stacks.write(f"Started {datetime.now(timezone.utc).isoformat()}; Python stacks every 120 seconds.\n")
        stacks.flush()
        print(f"[DataGen] Python stack diagnostics: {stack_path} (every 120s)", flush=True)
        faulthandler.dump_traceback_later(120, repeat=True, file=stacks, exit=False)
        try:
            return _collect(args, config, source_envset)
        except BaseException:
            traceback.print_exc(file=stacks)
            stacks.flush()
            raise
        finally:
            # Cancel before closing the descriptor, including interruption/failure.
            faulthandler.cancel_dump_traceback_later()


def _phase(args, name, callback):
    def record(state):
        message = f"{datetime.now(timezone.utc).isoformat()} [DataGen stage] {name}: {state}"
        print(message, flush=True)
        with (args.output / "progress.log").open("a", encoding="utf-8") as progress:
            progress.write(message + "\n")

    record("begin")
    started = monotonic()
    try:
        result = callback()
    except BaseException as exc:
        record(f"failed after {monotonic() - started:.1f}s ({type(exc).__name__})")
        raise
    record(f"done in {monotonic() - started:.1f}s")
    return result


def _collect(args, config, source_envset):
    # All runtime-heavy imports occur after SimulationApp initialization.
    from OmniNavExt.envset.core import SimulationBootstrap, SimulationConfig

    bootstrap = SimulationBootstrap(SimulationConfig(headless=args.headless))
    try:
        app = _phase(args, "SimulationApp startup", bootstrap.initialize)
        from OmniNavExt import import_extensions
        import_extensions()
        from OmniNav.core.task_config_manager.base import create_task_config_manager
        from OmniNavExt.envset.core import PhysicsManager, SceneManager
        from OmniNavExt.envset.core.scene_manager import is_matterport_scenario
        from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime
        from OmniNavExt.envset.world_utils import bootstrap_world_if_needed
        from bench.replay.replay_runner import ReplayConfig, ReplayRunner, ScenarioRef
        from datagen.pipeline import DataGenerationPipeline
        import omni.timeline

        scenario = json.loads(source_envset.read_text(encoding="utf-8"))["scenarios"][0]
        scene = scenario.get("scene") or {}
        navmesh = scenario.get("navmesh") or {}
        is_mp = is_matterport_scenario(scene)
        ref = ScenarioRef(source_envset, scenario, args.scenario, is_mp)
        replay = ReplayRunner(ReplayConfig(
            uninav_config=args.config, envset_path=source_envset, output_dir=args.output,
            scene_root=args.scene_root, headless=args.headless, fps=config.capture.fps,
            save_depth=True, num_cameras=1, skip_completed=False,
        ))
        model = replay._build_group_config_model([ref])
        runner = replay._create_runner_with_app(model, create_task_config_manager(model), app)
        runner._pre_physics_hook = lambda stage: replay._apply_sdf_to_scene_meshes(
            stage, scene.get("root_prim_path", "/World"), log_prefix="[DataGen SDF]")
        bootstrap_world_if_needed()
        EnvsetTaskRuntime.reset_navmesh_cache()
        if is_mp:
            prim = SceneManager.import_matterport_scene(scene, args.scene_root)
            SceneManager.prepare_matterport_navmesh(prim, navmesh, scene, app, envset_cfg=scenario)
            replay._set_camera_light()
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            timeline.pause()
        EnvsetTaskRuntime.reset_episode_state(stage=runner._stage)
        _phase(args, "runner.reset / scene and physics initialization",
               lambda: runner.reset(start_timeline=False))
        _phase(args, "NavMesh setup",
               lambda: replay._setup_first_scenario_navmesh(runner, scenario, args.scenario, scene, navmesh, is_mp, app))
        _phase(args, "timeline.play", timeline.play)
        EnvsetTaskRuntime.reconcile_virtual_humans(scenario, stage=runner._stage, defer_routes=True)
        EnvsetTaskRuntime.register_robots_as_dynamic_obstacles(runner)
        if not _phase(args, "wait for robot articulations", lambda: PhysicsManager.wait_for_articulations(
                runner, max_frames=replay._ARTICULATION_WAIT_MAX_FRAMES)):
            raise RuntimeError("Robot articulation initialization timed out")
        runner.render_interval = 0
        runner.render_trigger = 0
        _phase(args, "physics warmup",
               lambda: runner.warm_up(steps=replay._WARMUP_PHYSICS_STEPS, render=False, physics=True))
        _phase(args, "render warmup",
               lambda: runner.warm_up(steps=replay._WARMUP_RENDER_STEPS, render=True, physics=False))
        return _phase(args, "sampling and collection",
                      lambda: DataGenerationPipeline(config).run_with_runner(runner, scenario))
    finally:
        _phase(args, "SimulationApp shutdown", bootstrap.shutdown)
