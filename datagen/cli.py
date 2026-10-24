"""Collection CLI; help and preflight do not start or import Isaac Sim."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import sys
from uuid import uuid4

from datagen.config import CaptureConfig, NavMeshConfig, PipelineConfig, RobotConfig, TaskConfig


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Robot/physics YAML config")
    parser.add_argument("--envset", type=Path, required=True, help="Existing training scene envset JSON")
    parser.add_argument("--scenario", help="Required if the envset contains multiple scenarios")
    parser.add_argument("--scene-root", type=Path, help="Assets root; defaults to OMNINAV_SCENE_ROOT")
    parser.add_argument("--output", type=Path, required=True, help="Dataset directory; existing samples are preserved and new samples appended")
    parser.add_argument("--num-episodes", type=positive_int, default=1, help="Number of sampling attempts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-subtasks", type=positive_int, default=3,
                        help="Number of sequential ObjectNav goals; incomplete chains are rejected")
    parser.add_argument("--distinct-categories", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--room-zoning", action=argparse.BooleanOptionalAction, default=False,
                        help="Infer room names by object clustering (expensive for large scenes); off by default")
    parser.add_argument("--scene-id", help="Scene directory name; defaults to the source annotation's parent")
    parser.add_argument("--dataset-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--task", choices=["objectnav", "pointnav"], default="objectnav",
                        help="pointnav samples free reachable endpoints and geometry-based route instructions")
    parser.add_argument("--path-min-m", type=positive_float, default=2.0)
    parser.add_argument("--path-max-m", type=positive_float, default=10.0)
    parser.add_argument("--fps", type=positive_int, default=15)
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-clearance-m", type=positive_float, default=0.35)
    parser.add_argument("--navmesh-agent-radius-m", type=positive_float,
                        help="Override source NavMesh agent radius in meters")
    parser.add_argument("--navmesh-max-step-height-m", type=positive_float,
                        help="Override source NavMesh maximum step height in meters")
    parser.add_argument("--start-min-dist-m", type=positive_float, default=1.0)
    parser.add_argument("--object-start-min-dist-m", type=positive_float, default=2.0)
    parser.add_argument("--object-goal-min-m", type=positive_float, default=0.5)
    parser.add_argument("--object-goal-max-m", type=positive_float, default=1.5)
    parser.add_argument("--goal-threshold-m", type=positive_float, default=0.2)
    parser.add_argument("--min-visible-pixels", type=positive_int, default=100)
    parser.add_argument("--object-annotations", type=Path, help="Optional object IDs/categories/positions/prim paths JSON")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight", action="store_true", help="Check inputs and Python dependencies without launching Sim")
    args = parser.parse_args(argv)
    if args.object_goal_min_m > args.object_goal_max_m:
        parser.error("--object-goal-min-m must not exceed --object-goal-max-m")
    if args.path_min_m > args.path_max_m:
        parser.error("--path-min-m must not exceed --path-max-m")
    return args


def prepare_inputs(args):
    from OmniNav.local_paths import apply_local_path_env, resolve_scene_root

    apply_local_path_env()
    root = resolve_scene_root(args.scene_root, required=False)
    if root is None:
        raise ValueError("Set --scene-root or OMNINAV_SCENE_ROOT to the downloaded Assets directory")
    args.scene_root = root.expanduser().resolve()
    if not args.scene_root.is_dir():
        raise ValueError(f"Assets directory not found: {args.scene_root}")
    args.config = args.config.expanduser().resolve()
    args.envset = args.envset.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    for path in (args.config, args.envset, args.object_annotations):
        if path is not None and not path.expanduser().is_file():
            raise ValueError(f"Input file not found: {path}")
    if args.output.exists() and not args.output.is_dir():
        raise ValueError(f"Output must be a directory: {args.output}")
    raw = json.loads(args.envset.read_text(encoding="utf-8"))
    scenarios = raw.get("scenarios", [])
    if not scenarios:
        raise ValueError("envset must contain a non-empty scenarios list")
    if args.scenario is None and len(scenarios) != 1:
        raise ValueError("Multiple scenarios found; choose one with --scenario")
    selected = [s for s in scenarios if args.scenario is None or s.get("id") == args.scenario]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one matching scenario: {args.scenario}")
    scenario = selected[0]
    from datagen.export.standard import safe_component, robot_directory
    args.scene_id = safe_component(args.scene_id or args.envset.parent.name)
    args.robot_directory = robot_directory(scenario)
    args.scenario = scenario.get("id") or "datagen_source"
    scenario["id"] = args.scenario
    if len((scenario.get("robots") or {}).get("entries") or []) != 1:
        raise ValueError("Collection currently requires exactly one robot in robots.entries")
    units = float((scenario.get("scene") or {}).get("units_in_meters", 0))
    if not math.isfinite(units) or units <= 0:
        raise ValueError("scene.units_in_meters must explicitly specify a positive finite scale")
    # The scene loaders multiply these envset values by units_in_meters before baking.
    for field, meters in (("agent_radius", args.navmesh_agent_radius_m),
                          ("max_step_height", args.navmesh_max_step_height_m)):
        if meters is not None:
            scenario.setdefault("navmesh", {})[field] = meters / units
    pose = scenario["robots"]["entries"][0].get("initial_pose") or {}
    position = pose.get("position")
    if not isinstance(position, list) or len(position) != 3 or not all(math.isfinite(float(v)) for v in position):
        raise ValueError("Robot initial_pose.position must contain three finite coordinates (including spawn height)")
    if not math.isfinite(float(pose.get("orientation_deg", float("nan")))):
        raise ValueError("Robot initial_pose.orientation_deg must be finite")
    # Reuse assets/spawn/navmesh, but never carry old tasks or expert demonstrations into new samples.
    scenario["task"] = {"navigation": {}}
    scenario.pop("recording", None)
    scenario.pop("answer", None)
    scenario.pop("datagen", None)
    scenario.pop("datagen_segments", None)
    scenario.pop("datagen_endpoint_task", None)
    scenario.pop("virtual_humans", None)  # first-stage ObjectNav collection is static
    scenario["robots"]["entries"][0].pop("rb_gt_waypoints", None)

    from OmniNavExt.envset.config_loader import EnvsetConfigLoader
    EnvsetConfigLoader.normalize_scenario_paths(scenario, args.scene_root)
    import yaml
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dt = float((base.get("simulator") or {}).get("physics_dt", 1 / 60))
    if not math.isfinite(dt) or dt <= 0 or args.fps > 1 / dt + 1e-5:
        raise ValueError("Capture fps must not exceed the positive physics step frequency")
    cfg = PipelineConfig(
        enable_room_zoning=args.room_zoning,
        output_dir=args.output,
        navmesh=NavMeshConfig(min_clearance_m=args.min_clearance_m),
        robot=RobotConfig(),
        task=TaskConfig(
            types=["ObjectNav" if args.task == "objectnav" else "PointNav"],
            chain_length=args.num_subtasks if args.task == "objectnav" else 1, require_full_chain=True,
            distinct_object_targets=True, distinct_object_categories=args.distinct_categories,
            sample_endpoints=args.task == "pointnav", path_min_m=args.path_min_m, path_max_m=args.path_max_m,
            num_episodes=args.num_episodes, random_start=args.random_start,
            start_min_dist_m=args.start_min_dist_m,
            object_start_min_dist_m=args.object_start_min_dist_m,
            object_goal_min_m=args.object_goal_min_m, object_goal_max_m=args.object_goal_max_m,
            final_goal_threshold_m=args.goal_threshold_m,
        ),
        capture=CaptureConfig(fps=args.fps), random_seed=args.seed,
        object_annotations_path=args.object_annotations.expanduser().resolve() if args.object_annotations else None,
        min_pixels_visible=args.min_visible_pixels if args.task == "objectnav" else 0,
    )
    return scenario, cfg


def main(argv=None):
    args = parse_args(argv)
    try:
        scenario, cfg = prepare_inputs(args)
        dataset = (args.dataset_output or args.output).expanduser().resolve()
        # Batch supplies its own unique working directory. Standalone runs share
        # the dataset root, but never reuse intermediate files from an earlier run.
        if args.dataset_output is None:
            args.output = dataset / "_runs" / uuid4().hex
        cfg.output_dir = args.output / "_episodes"
        missing = [name for name in ("isaacsim", "numpy", "cv2", "PIL", "pydantic")
                   if importlib.util.find_spec(name) is None]
        report = {"source_envset": str(args.envset), "scenario": args.scenario, "dataset_root": str(dataset),
                  "scene_root": str(args.scene_root), "python": sys.version.split()[0],
                  "missing_modules": missing, "pipeline": asdict(cfg)}
        if args.preflight:
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            return 2 if missing else 0
        if missing:
            raise ValueError(f"Missing runtime modules: {missing}. Use the project's Isaac Sim 5.0 / Python 3.11 environment.")
        args.output.mkdir(parents=True, exist_ok=False)
        print(f"Dataset: {dataset}\nRun diagnostics: {args.output}", flush=True)
        with (args.output / "collection.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, default=str)
        source = args.output / "source_envset.json"
        source.write_text(json.dumps({"scenarios": [scenario]}, indent=2, ensure_ascii=False), encoding="utf-8")
        from datagen.runtime import collect
        try:
            episodes = collect(args, cfg, source)
            from datagen.training_samples import build_manifest
            summary = build_manifest(cfg.output_dir, episodes, requested=args.num_episodes)
            from datagen.export.standard import publish_samples
            summary = publish_samples(
                cfg.output_dir, dataset,
                scene_id=args.scene_id, robot=args.robot_directory, summary=summary,
                manifest_path=args.output / "manifest.jsonl",
            )
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except Exception as exc:
            (args.output / "summary.json").write_text(
                json.dumps({"status": "failed", "error": str(exc)}, indent=2), encoding="utf-8")
            raise
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary["accepted"] == args.num_episodes else 2
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"Data generation failed: {exc}", file=sys.stderr)
        return 2
