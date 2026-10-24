"""Publish validated episodes using OmniNavBench's annotation/video directory layout."""
import json
import fcntl
import re
from pathlib import Path
import shutil


def safe_component(value):
    value = str(value)
    if not value or value in {".", ".."} or any(c in value for c in ("/", "\\", "\x00")):
        raise ValueError(f"Unsafe dataset path component: {value!r}")
    return value


def robot_directory(scenario):
    entries = (scenario.get("robots") or {}).get("entries") or []
    if len(entries) != 1:
        raise ValueError("Collection requires exactly one robot")
    name = str(entries[0].get("type", "")).lower()
    mapping = {"carter_v1": "car", "carter": "car", "aliengo": "dog", "h1": "human"}
    if name not in mapping:
        raise ValueError(f"Unsupported robot embodiment for standard export: {name}")
    return mapping[name]


def publish_sample(work, dataset, *, scene_id, robot, index):
    scene_id, robot = safe_component(scene_id), safe_component(robot)
    stem = f"final_episode_{int(index)}"
    annotation_rel = Path("annotations/train/original") / robot / scene_id / f"{stem}.json"
    video_rel = Path("videos/train") / robot / scene_id / stem
    annotation, video = dataset / annotation_rel, dataset / video_rel
    if annotation.exists() or video.exists():
        raise FileExistsError(f"Refusing to overwrite published sample: {annotation}")
    payload = json.loads((work / "episode.json").read_text(encoding="utf-8"))
    scenario = payload["scenarios"][0]
    scenario["id"] = f"{robot}_{scene_id}_{stem}"
    media = {"rgb": str(video_rel / "rgb.mp4"), "depth": str(video_rel / "depth.mp4"),
             "trajectory": str(video_rel / "path.json"),
             "trajectory_targets": str(video_rel / "trajectory_targets.jsonl"),
             "path_base": "dataset_root", "frame_axis": "zero_based_video_frame",
             "depth_video_encoding": "8-bit clipped visualization; use raw depth for quantitative training"}
    if (work / "video/depth_raw").is_dir():
        media["depth_raw"] = str(video_rel / "depth_raw")
        media["depth_raw_encoding"] = "float32 npy; unchanged distance_to_image_plane annotator output"
        media["depth_invalid_mask"] = "isfinite(depth) & (depth > 0)"
    scenario["media"] = media
    recording = scenario["recording"]
    scenario["robots"]["entries"][0]["rb_gt_waypoints"] = recording["gt_path"]
    video.parent.mkdir(parents=True, exist_ok=True)
    video.mkdir()  # exclusive ownership; never overwrite another collector's sample
    annotation_created = False
    try:
        shutil.copytree(work / "video", video, dirs_exist_ok=True)
        shutil.copy2(work / "path/path.json", video / "path.json")
        shutil.copy2(work / "trajectory_targets.jsonl", video / "trajectory_targets.jsonl")
        annotation.parent.mkdir(parents=True, exist_ok=True)
        with annotation.open("x", encoding="utf-8") as handle:
            annotation_created = True
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    except Exception:
        shutil.rmtree(video)
        if annotation_created:
            annotation.unlink(missing_ok=True)
        raise
    result = {"sample_id": scenario["id"], "scenario_id": scenario["id"],
              "episode": str(annotation_rel), "rgb_video": media["rgb"], "depth_video": media["depth"],
              "trajectory": media["trajectory"], "trajectory_targets": media["trajectory_targets"],
              "scene_id": scene_id, "robot": robot, "num_subtasks": len(scenario["task"].get("subtasks", []))}
    if "depth_raw" in media:
        result["depth_raw"] = media["depth_raw"]
    return result


def next_episode_index(dataset, scene_id, robot):
    """Respect both complete samples and media left by interrupted publications."""
    indexes = []
    for directory in (dataset / "annotations/train/original" / robot / scene_id,
                      dataset / "videos/train" / robot / scene_id):
        if directory.is_dir():
            for path in directory.iterdir():
                match = re.fullmatch(r"final_episode_(\d+)(?:\.json)?", path.name)
                if match:
                    indexes.append(int(match.group(1)))
    return max(indexes, default=-1) + 1


def publish_samples(work, dataset, *, scene_id, robot, summary, manifest_path):
    scene_id, robot = safe_component(scene_id), safe_component(robot)
    records = [json.loads(line) for line in (work / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    published, errors = [], []
    dataset.mkdir(parents=True, exist_ok=True)
    root_manifest = dataset / "manifest.jsonl"
    # Serialize number allocation, publication and index append across collectors.
    with (dataset / ".publish.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        index = next_episode_index(dataset, scene_id, robot)
        with root_manifest.open("a", encoding="utf-8") as handle:
            for record in records:
                source = work / record["episode"]
                try:
                    paths = publish_sample(source.parent, dataset, scene_id=scene_id, robot=robot, index=index)
                except (ValueError, OSError) as exc:
                    errors.append({"sample": record["sample_id"], "error": str(exc)})
                else:
                    entry = {**record, **paths}
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    handle.flush()
                    published.append(entry)
                index += 1
    if manifest_path.resolve() != root_manifest.resolve():
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in published:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {**summary, "accepted": len(published), "publication_failures": errors,
            "dataset_root": str(dataset),
            "status": "complete" if len(published) == summary["requested_attempts"] else "partial" if published else "failed"}
