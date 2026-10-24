"""Validate actual decoded frames against trajectory samples before admitting data."""
from __future__ import annotations

import json
import math
from pathlib import Path


def inspect_video(path: Path) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Video cannot be opened: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        advertised = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        shape = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise ValueError(f"Empty video frame: {path}")
            if shape is not None and shape != frame.shape:
                raise ValueError(f"Video resolution changes: {path}")
            shape = frame.shape
            count += 1
        if count < 2 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video has insufficient frames or invalid fps: {path}")
        if advertised > 0 and count != advertised:
            raise ValueError(f"Video is not fully decodable: {path}: {count}/{advertised}")
        return {"frames": count, "fps": fps, "width": shape[1], "height": shape[0]}
    finally:
        capture.release()


def validate_sample(directory: Path) -> dict:
    payload = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    scenario = payload["scenarios"][0]
    recording = json.loads((directory / "path/path.json").read_text(encoding="utf-8"))
    if recording != scenario.get("recording"):
        raise ValueError("Embedded and sidecar recordings disagree")
    legacy = ((scenario.get("robots") or {}).get("entries") or [{}])[0].get("rb_gt_waypoints")
    if legacy is not None and legacy != recording.get("gt_path"):
        raise ValueError("Legacy and canonical GT disagree")
    points = recording.get("gt_path") or []
    if len(points) < 2:
        raise ValueError("Fewer than two trajectory samples")
    previous_time = -math.inf
    for index, point in enumerate(points):
        if point.get("frame") != index:
            raise ValueError("Trajectory frame indices must match zero-based video frames")
        values = list(point["xyz"]) + [point["yaw_deg"], point["time_s"]]
        if len(point["xyz"]) != 3 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Invalid pose or timestamp")
        if point["time_s"] <= previous_time:
            raise ValueError("Trajectory timestamps must increase strictly")
        previous_time = point["time_s"]
    rgb = inspect_video(directory / "video/rgb.mp4")
    depth = inspect_video(directory / "video/depth.mp4")
    if rgb["frames"] != len(points) or depth["frames"] != len(points):
        raise ValueError("RGB/depth frame counts do not match trajectory samples")
    if abs(rgb["fps"] - depth["fps"]) > 1e-3:
        raise ValueError("RGB/depth video fps disagree")
    frame_subtasks = validate_segments(scenario, points)
    if (recording.get("metadata") or {}).get("raw_depth"):
        import numpy as np
        depth_files = sorted((directory / "video/depth_raw").glob("*.npy"))
        if [p.name for p in depth_files] != [f"{i:06d}.npy" for i in range(len(points))]:
            raise ValueError("Raw depth frames do not match trajectory/video frames")
        for path in depth_files:
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            if array.dtype != np.float32 or array.shape != (rgb["height"], rgb["width"]):
                raise ValueError(f"Invalid raw depth dtype/shape: {path}")
    instruction = scenario["task"]["navigation"].get("instruction")
    if not instruction:
        raise ValueError("Missing instruction")
    from datagen.trajectory_targets import write_targets
    write_targets(directory / "trajectory_targets.jsonl", points, float(scenario["scene"]["units_in_meters"]),
                  frame_subtasks=frame_subtasks)
    return {
        "sample_id": directory.name, "scenario_id": scenario["id"], "instruction": instruction,
        "episode": f"{directory.name}/episode.json", "trajectory": f"{directory.name}/path/path.json",
        "trajectory_targets": f"{directory.name}/trajectory_targets.jsonl",
        "rgb_video": f"{directory.name}/video/rgb.mp4", "depth_video": f"{directory.name}/video/depth.mp4",
        "frames": len(points), "fps": rgb["fps"], "width": rgb["width"], "height": rgb["height"],
        "simulation_duration_s": points[-1]["time_s"] - points[0]["time_s"],
        "units_in_meters": scenario["scene"]["units_in_meters"],
        "depth_encoding": "8-bit visualization, clipped to 0..10m; not metric depth",
        "command_source": "pose-derived normalized velocity estimates; not executed controller actions",
    }


def validate_segments(scenario, points):
    metadata = (scenario.get("recording") or {}).get("metadata") or {}
    segments = scenario.get("datagen_segments") or []
    object_task = any(str(t).lower() in {"objectnav", "objnav"} for t in metadata.get("task_types", []))
    if not segments and not object_task:
        return None
    expected = int(metadata.get("requested_subtasks", len(segments)))
    task = scenario["task"]
    subtasks, instructions = task.get("subtasks", []), task.get("sub_instructions", [])
    if expected < 1 or len(segments) != expected or len(subtasks) != expected or len(instructions) != expected:
        raise ValueError("Incomplete ObjectNav task chain")
    mapping, ids, categories = [], [], []
    for i, segment in enumerate(segments):
        start, end = segment["start_capture_frame"], segment["end_capture_frame"]
        if segment["step"] != i or start != len(mapping) or end < start or end >= len(points):
            raise ValueError("Invalid or noncontiguous subtask frame boundaries")
        target = segment["target"]
        if subtasks[i].get("type") != "GOTO_OBJECT" or subtasks[i].get("object_id") != target["object_id"]:
            raise ValueError("Subtask target mismatch")
        if instructions[i].get("text") != segment["instruction"] or instructions[i].get("step") != i:
            raise ValueError("Subtask instruction mismatch")
        if task["navigation"]["objects"].get(target["object_id"]) != target["position"]:
            raise ValueError("Target position mismatch")
        if segment["start_time_s"] != points[start]["time_s"] or segment["end_time_s"] != points[end]["time_s"]:
            raise ValueError("Subtask frame/time mismatch")
        ids.append(target["object_id"])
        categories.append(str(target["category"]).lower())
        mapping.extend([i] * (end-start+1))
    if len(mapping) != len(points):
        raise ValueError("Subtasks do not cover the entire recording")
    if metadata.get("distinct_object_targets") and len(set(ids)) != len(ids):
        raise ValueError("Repeated ObjectNav instance")
    if metadata.get("distinct_object_categories") and len(set(categories)) != len(categories):
        raise ValueError("Repeated ObjectNav category")
    return mapping


def build_manifest(output: Path, episodes, *, requested: int) -> dict:
    accepted, rejected = [], []
    for episode in episodes:
        try:
            accepted.append(validate_sample(Path(episode.output_dir)))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            rejected.append({"episode": str(episode.output_dir), "error": str(exc)})
    failures = []
    for path in sorted(output.glob("episode_*/attempt.json")):
        attempt = json.loads(path.read_text(encoding="utf-8"))
        if attempt.get("status") != "succeeded":
            failures.append({"episode": path.parent.name, "error": attempt.get("error", "incomplete")})
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for entry in accepted:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    summary = {"status": "complete" if len(accepted) == requested else "partial" if accepted else "failed",
               "requested_attempts": requested, "generated": len(episodes), "accepted": len(accepted),
               "generation_failures": failures, "validation_failures": rejected}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
