"""NavMesh endpoint sampling and geometry-grounded route instructions."""
import math
import numpy as np


def sample_endpoint_path(geometry, start, *, units, min_length_m, max_length_m,
                         attempts=100, min_clearance_m=0.0):
    start = np.asarray(start, dtype=np.float32)
    for _ in range(attempts):
        goal = geometry.sample_random_point()
        if goal is None:
            continue
        goal = np.asarray(goal, dtype=np.float32)
        # This executor is planar: reject stairs/multi-floor paths.
        path = geometry.query_path(start, goal)
        if path is None or len(path) < 2:
            continue
        points = [np.asarray(p, dtype=np.float32) for p in path]
        if np.linalg.norm((points[0] - start)[:2]) * units > 0.1:
            continue
        if (max(p[2] for p in points) - min(p[2] for p in points)) * units > 0.1:
            continue
        length = sum(float(np.linalg.norm((b - a)[:2])) for a, b in zip(points, points[1:])) * units
        if not min_length_m <= length <= max_length_m:
            continue
        if min_clearance_m > 0 and geometry.get_clearance(goal) * units < min_clearance_m:
            continue
        return points, length
    raise RuntimeError(f"No reachable same-floor endpoint with path length {min_length_m}..{max_length_m}m in {attempts} attempts")


def route_instruction(points, yaw_rad, units):
    """Describe every nonzero NavMesh leg; no invented semantic landmarks."""
    parts = []
    heading = float(yaw_rad)
    for a, b in zip(points, points[1:]):
        dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
        distance = math.hypot(dx, dy) * units
        if distance < 1e-6:
            continue
        angle = math.atan2(dy, dx)
        turn = (angle - heading + math.pi) % (2 * math.pi) - math.pi
        degrees = abs(math.degrees(turn))
        if degrees >= 5:
            parts.append(f"Turn {'left' if turn > 0 else 'right'} approximately {degrees:.0f} degrees.")
        parts.append(f"Move forward approximately {distance:.2f} meters.")
        heading = angle
    if not parts:
        raise ValueError("Cannot describe an empty route")
    return " ".join(parts + ["Stop at the destination."])
