# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Open-loop replay of stored command trajectories (used to execute RoboFS-T2A predictions)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class ReplayPolicy:
    """Return, for each environment, the stored command of its word at the next control step's target time.

    During the preparation hold the first command is sent; after the last stored frame the last one is held.

    Args:
        env: Unwrapped manager-based environment whose motion command selects the clips.
        replay_file: NPZ with ``actions`` [N, T, 18] (normalized, 30 Hz), ``lengths`` [N] and ``words`` [N].
    """

    method = "replay"

    def __init__(self, env, replay_file: Path):
        self.env = env
        self.term = env.command_manager.get_term("motion")
        with np.load(replay_file, allow_pickle=False) as data:
            actions, lengths, words = data["actions"], data["lengths"], [str(w) for w in data["words"]]
        self.actions = torch.as_tensor(actions, dtype=torch.float32, device=env.device).clamp(-1, 1)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long, device=env.device)
        self.rows = {word: index for index, word in enumerate(words)}
        self._motion, self._clip_rows = None, None

    def _rows(self) -> torch.Tensor:
        """Map the command's clip ids to replay rows by word, following the evaluated motion."""
        motion = self.term.motion
        if motion is not self._motion:
            missing = [w for w in motion.words if w not in self.rows]
            if missing:
                raise KeyError(f"Replay file lacks {len(missing)} evaluated words, e.g. {missing[0]!r}.")
            self._clip_rows = torch.tensor([self.rows[w] for w in motion.words], device=self.env.device)
            self._motion = motion
        return self._clip_rows[self.term.clip_ids]

    def __call__(self, observations) -> torch.Tensor:
        term, dt = self.term, self.env.step_dt
        # The command computed now is judged against the next step's target.
        times = term.reference_time + dt
        motion_times = (times - term.cfg.preparation_duration_s).clamp(min=0) * term.speed
        rows = self._rows()
        frames = torch.round(motion_times * term.motion.fps).long()
        frames = torch.minimum(frames, self.lengths[rows] - 1)
        return self.actions[rows, frames]
