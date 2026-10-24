"""Sample new trajectories once per GRScenes asset, not once per source episode/style."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


def discover_scenes(dataset_root, robot="car"):
    root = Path(dataset_root) / "annotations/train/original" / robot
    if not root.is_dir():
        raise ValueError(f"Training annotation directory missing: {root}")
    unique = {}
    for path in sorted(root.rglob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for scenario in raw.get("scenarios", []):
            scene = scenario.get("scene") or {}
            label = str(scene.get("category", "")) + " " + str(scene.get("type", ""))
            asset = scene.get("usd_path") or scene.get("asset_path")
            if scene.get("use_matterport") or scene.get("matterport") or "matterport" in label.lower() or "mp3d" in label.lower():
                continue
            if not asset or "grscene" not in (label + " " + str(asset)).lower():
                continue
            if not scenario.get("id"):
                raise ValueError(f"Scenario without ID: {path}")
            key = json.dumps([asset, scene.get("units_in_meters"), scene.get("root_prim_path")])
            digest = hashlib.sha256(key.encode()).hexdigest()[:12]
            unique.setdefault(key, {"scene_key": digest, "scene_id": path.parent.name, "asset": str(asset),
                                    "envset": str(path.resolve()), "scenario": scenario["id"]})
    if not unique:
        raise ValueError("No explicitly identified GRScenes scenarios found in original training annotations")
    return list(unique.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", choices=["car", "dog", "human"], default="car")
    parser.add_argument("--episodes-per-scene", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=1, help="0 = all; default is a one-scene smoke run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene-timeout-s", type=int, default=3600)
    parser.add_argument("--plan-only", action="store_true", help="Print deduplicated plan without creating output or starting Sim")
    args, extra = parser.parse_known_args(argv)
    if args.episodes_per_scene < 1 or args.max_scenes < 0 or args.scene_timeout_s <= 0:
        parser.error("episodes/timeout must be positive; max-scenes must be nonnegative")
    # Sampling flags only: never let passthrough override a scene's identity or output.
    from datagen.cli import parse_args
    forbidden = {"--envset", "--scenario", "--num-episodes", "--task", "--preflight", "--scene-id", "--dataset-output"}
    if any(token.split("=")[0] in forbidden for token in extra):
        parser.error("envset/scenario/num-episodes/task/preflight are managed by the batch runner")
    try:
        scenes = discover_scenes(args.dataset_root, args.robot)
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]
        if len({s["scene_id"] for s in scenes}) != len(scenes):
            raise ValueError("Different scene assets share a directory name; resolve source scene IDs before exporting")
        output = args.output.expanduser().resolve()
        run_dir = output / "_runs" / ("batch_" + uuid4().hex)
        plan = []
        for scene in scenes:
            # Stable across max-scenes and source annotation ordering.
            seed = (args.seed + int(scene["scene_key"], 16)) % (2**31)
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "runDataGen.py"),
                       "--config", str(args.config.expanduser().resolve()),
                       "--envset", scene["envset"], "--scenario", scene["scenario"],
                       "--scene-root", str(args.scene_root.expanduser().resolve()),
                       "--output", str(run_dir / "_work" / scene["scene_key"]), "--task", "objectnav",
                       "--dataset-output", str(output), "--scene-id", scene["scene_id"],
                       "--num-episodes", str(args.episodes_per_scene), "--seed", str(seed), *extra]
            parse_args(command[2:])  # reject typos before starting any expensive scene
            plan.append({**scene, "seed": seed, "command": command})
        if args.plan_only:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        if output.exists() and not output.is_dir():
            raise ValueError(f"Output must be a directory: {output}")
        run_dir.mkdir(parents=True, exist_ok=False)
        print(f"Dataset: {output}\nBatch diagnostics: {run_dir}", flush=True)
        (run_dir / "batch_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        results = []
        with (run_dir / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
            for entry in plan:
                key = entry["scene_key"]
                print(f"Collecting GRScenes asset {entry['asset']} -> {key}", flush=True)
                with (run_dir / f"{key}.log").open("w") as log:
                    try:
                        process = subprocess.run(entry["command"], stdout=log, stderr=subprocess.STDOUT,
                                                 timeout=args.scene_timeout_s, check=False)
                        code = process.returncode
                    except subprocess.TimeoutExpired:
                        code = 124
                count = 0
                child_manifest = run_dir / "_work" / key / "manifest.jsonl"
                if child_manifest.is_file():
                    for line in child_manifest.read_text(encoding="utf-8").splitlines():
                        record = json.loads(line)
                        record["scene_key"] = key
                        record["source_scene_asset"] = entry["asset"]
                        manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
                    manifest.flush()
                results.append({"scene_key": key, "returncode": code, "accepted": count})
                (run_dir / "batch_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        return 0 if all(r["returncode"] == 0 for r in results) else 2
    except (ValueError, OSError) as exc:
        print(f"Batch failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
