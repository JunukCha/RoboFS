# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Palm alignment: convert OpenFS landmarks to approximate palm-local positions [m]."""

from __future__ import annotations

import numpy as np

# Wrist-to-middle-MCP length [m] the source hand is scaled to.
PALM_LENGTH_M = 0.10


def align_palm(raw: np.ndarray, depth_sign: int = 1) -> np.ndarray:
    """Express non-metric OpenFS XYZ landmarks in a palm-local frame [m].

    Each frame is rotated into its own MCP palm basis (z: wrist to middle MCP, x: little MCP to index MCP,
    orthogonalized), and the clip is scaled so that its median wrist-to-middle-MCP length is 0.10 m. The
    image depth axis is treated as having the same scale as the image plane.

    Args:
        raw: OpenFS landmarks, shape [T, 21, 3], MediaPipe order.
        depth_sign: +1 or -1, a reflection of the depth axis before alignment.

    Returns:
        Palm-local landmarks [m], shape [T, 21, 3], float32; landmark 0 (wrist) is the origin.
    """
    if raw.ndim != 3 or raw.shape[1:] != (21, 3) or not np.isfinite(raw).all() or depth_sign not in (-1, 1):
        raise ValueError("Expected finite [T, 21, 3] poses and depth_sign +/-1.")
    centered = raw.astype(np.float64) - raw[:, :1]
    centered[..., 2] *= depth_sign
    z = centered[:, 9]
    scale = np.linalg.norm(z, axis=-1)
    if np.any(scale < 1e-7):
        raise ValueError("Degenerate wrist-to-middle-MCP direction.")
    z = z / scale[:, None]
    x = centered[:, 5] - centered[:, 17]
    x -= (x * z).sum(axis=-1, keepdims=True) * z
    norm = np.linalg.norm(x, axis=-1)
    if np.any(norm < 1e-7):
        raise ValueError("Degenerate palm basis.")
    x /= norm[:, None]
    y = np.cross(z, x)
    basis = np.stack((x, y, z), axis=-1)
    local = np.einsum("tji,tik->tjk", centered, basis) * (PALM_LENGTH_M / np.median(scale))
    return local.astype(np.float32)
