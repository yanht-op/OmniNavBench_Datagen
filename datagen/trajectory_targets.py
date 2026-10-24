"""Derive fixed-time future positions in the current robot's local frame."""
import json
import math
import numpy as np


def future_targets(points, units, horizons=(0.5, 1.0, 1.5, 2.0)):
    times = np.asarray([p["time_s"] for p in points], dtype=float)
    xy = np.asarray([p["xyz"][:2] for p in points], dtype=float) * units
    if len(times) < 2 or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("Future targets require finite positions and increasing timestamps")
    for i, point in enumerate(points):
        yaw = math.radians(float(point["yaw_deg"]))
        c, s = math.cos(yaw), math.sin(yaw)
        values, valid = [], []
        for offset in horizons:
            target_time = times[i] + offset
            available = target_time <= times[-1] + 1e-9
            valid.append(bool(available))
            if not available:
                values.append([0.0, 0.0])  # explicitly masked, never extrapolated beyond episode
                continue
            x = float(np.interp(target_time, times, xy[:, 0]) - xy[i, 0])
            y = float(np.interp(target_time, times, xy[:, 1]) - xy[i, 1])
            values.append([c*x + s*y, -s*x + c*y])
        yield {"frame": point["frame"], "time_s": point["time_s"],
               "future_offsets_s": list(horizons), "future_xy_local_m": values,
               "valid": valid, "is_last_observation": i == len(points)-1}


def write_targets(path, points, units, *, frame_subtasks=None):
    with path.open("w", encoding="utf-8") as handle:
        for i, target in enumerate(future_targets(points, units)):
            if frame_subtasks is not None:
                target["subtask_index"] = frame_subtasks[i]
                target["subtask_end"] = i == len(points)-1 or frame_subtasks[i+1] != frame_subtasks[i]
            handle.write(json.dumps(target) + "\n")
