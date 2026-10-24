"""Lossless primary-camera depth, separate from the lossy preview video."""
import numpy as np


def save_depth_frame(directory, frame, depth):
    if depth is None:
        raise RuntimeError("Missing depth annotation for captured RGB frame")
    array = np.asarray(depth, dtype=np.float32)
    if array.ndim != 2 or not array.size:
        raise RuntimeError("Expected a nonempty HxW depth array")
    directory.mkdir(parents=True, exist_ok=True)
    # Preserve invalid returns (NaN/Inf/zero) so training can mask them explicitly.
    np.save(directory / f"{frame:06d}.npy", array, allow_pickle=False)
