# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""PPO runner that saves checkpoints only at interval boundaries, without modifying the installed RSL-RL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rsl_rl.runners import OnPolicyRunner


class HandCheckpointRunner(OnPolicyRunner):
    """Save ``model_<epoch>.pt`` only when the one-based completed epoch is a multiple of ``save_interval``."""

    def __init__(self, env, train_cfg: dict[str, Any], log_dir: str | None = None, device: str = "cpu"):
        self.checkpoint_interval = train_cfg["save_interval"]
        if self.checkpoint_interval <= 0:
            raise ValueError("save_interval must be positive.")
        self._last_saved_epoch: int | None = None
        # Let upstream call save each epoch; this runner owns the one-based trigger.
        super().__init__(env, {**train_cfg, "save_interval": 1}, log_dir=log_dir, device=device)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Restore the full training state and continue after the completed saved iteration."""
        infos = super().load(path, load_cfg, strict, map_location)
        if load_cfg is None:
            self.current_learning_iteration += 1
        return infos

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save only when the completed epoch is an interval multiple."""
        epoch = self.current_learning_iteration + 1
        if epoch % self.checkpoint_interval or epoch == self._last_saved_epoch:
            return
        super().save(str(Path(path).with_name(f"model_{epoch}.pt")), infos)
        self._last_saved_epoch = epoch
