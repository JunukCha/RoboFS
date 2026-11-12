# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Palm-local reference landmark sequences (a padded multi-clip dataset), independent of the simulator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# Fingertip landmarks in MediaPipe order: thumb, index, middle, ring, little.
TIPS = [4, 8, 12, 16, 20]
# Landmark chains of the five fingers (base joint to tip).
CHAINS = tuple(tuple(range(start, start + 4)) for start in (1, 5, 9, 13, 17))


class HandMotion:
    """Padded palm-local positions [m], shape [N, T, 21, 3], with per-clip lengths and words.

    Args:
        path: Dataset NPZ with ``poses`` [N, T, 21, 3], ``lengths`` [N], ``words`` [N] and JSON ``metadata``.
        device: Torch device of the stored tensors.
    """

    def __init__(self, path: str | Path, device: str):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            values = archive["poses"].copy()
            lengths = archive["lengths"].copy()
            self.words = archive["words"].tolist()
        if values.ndim != 4 or values.shape[2:] != (21, 3) or not len(values):
            raise ValueError("Expected dataset poses [N, T, 21, 3].")
        if (
            lengths.shape != (len(values),)
            or not np.issubdtype(lengths.dtype, np.integer)
            or np.any(lengths < 2)
            or np.any(lengths > values.shape[1])
            or len(self.words) != len(values)
        ):
            raise ValueError("Invalid dataset lengths or words.")
        if metadata.get("coordinate_frame") != "Shadow Hand rh_palm body-local":
            raise ValueError("Expected Shadow Hand rh_palm body-local coordinates.")
        if metadata.get("units") != "meters" or metadata.get("ordering") != "MediaPipe":
            raise ValueError("Expected meters and MediaPipe landmark order.")
        if not np.isfinite(values).all() or not np.allclose(values[..., 0, :], 0, atol=1e-6):
            raise ValueError("Landmarks must be finite and landmark 0 must be the palm origin.")
        self.fps = float(metadata["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Motion fps must be positive and finite.")
        self.points = torch.as_tensor(values, dtype=torch.float32, device=device)
        self.metadata = metadata
        self.lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)
        self.durations = (self.lengths - 1) / self.fps
        self.duration = (int(self.lengths.max()) - 1) / self.fps

    def sample(self, times: torch.Tensor, clip_ids: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate targets [m] at times [s], clamped to each clip's endpoints.

        Args:
            times: Clip times [s], shape [N].
            clip_ids: Clip index per time, shape [N].

        Returns:
            Landmark positions [m], shape [N, 21, 3].
        """
        if clip_ids.shape != times.shape:
            raise ValueError("Dataset sampling requires clip_ids with the same shape as times.")
        last = self.lengths[clip_ids] - 1
        frames = torch.minimum((times * self.fps).clamp(min=0), last)
        lower = frames.floor().long()
        upper = torch.minimum(lower + 1, last)
        alpha = (frames - lower)[..., None, None]
        return torch.lerp(self.points[clip_ids, lower], self.points[clip_ids, upper], alpha)
