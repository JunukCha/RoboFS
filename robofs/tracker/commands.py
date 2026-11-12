# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Reference motion command: palm-local targets after a preparation hold, and measured robot landmarks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

from robofs.tracker.motion import TIPS, HandMotion

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Joint-frame positions, not centers of mass. Thumb MCP is the middle-link origin.
LANDMARK_BODIES = ["rh_palm", "rh_thbase", "rh_thmiddle", "rh_thdistal", "rh_thdistal"] + [
    f"rh_{finger}{part}" for finger in ("ff", "mf", "rf", "lf") for part in ("proximal", "middle", "distal", "distal")
]


class HandMotionCommand(CommandTerm):
    """Serve reference landmarks after an initial hold of the first pose, and measure the robot's landmarks."""

    def __init__(self, cfg: HandMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not cfg.motion_file:
            raise ValueError("commands.motion.motion_file must point to a dataset NPZ.")
        self.robot = env.scene["robot"]
        self.body_ids = [self.robot.find_bodies(name)[0][0] for name in LANDMARK_BODIES]
        self.offsets = torch.zeros(21, 3, device=self.device)
        # Virtual axial tips [m], near mesh endpoints rather than contact patch centers.
        self.offsets[TIPS, 2] = torch.tensor([0.035, 0.032, 0.032, 0.032, 0.032], device=self.device)
        self.motion = HandMotion(cfg.motion_file, self.device)
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if cfg.speed_range[0] <= 0 or cfg.speed_range[1] < cfg.speed_range[0]:
            raise ValueError("speed_range must be ordered and positive.")
        if not np.isfinite(cfg.preparation_duration_s) or cfg.preparation_duration_s < 0:
            raise ValueError("preparation_duration_s must be finite and nonnegative.")
        self.speed = torch.ones(self.num_envs, device=self.device)
        self.metrics["landmark_error_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["fingertip_error_m"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def reference_time(self) -> torch.Tensor:
        """Elapsed per-environment episode time [s], including preparation."""
        return self._env.episode_length_buf * self._env.step_dt

    @property
    def target(self) -> torch.Tensor:
        """Current target positions [m], shape [N, 21, 3]."""
        return self.sample_target(self.reference_time)

    @property
    def reference_duration(self) -> torch.Tensor:
        """Playback duration [s] of each selected clip, including preparation."""
        return self.cfg.preparation_duration_s + self.motion.durations[self.clip_ids] / self.speed

    def sample_target(self, times: torch.Tensor) -> torch.Tensor:
        """Sample the target at episode times [s] (shape [N]); the first pose is held during preparation."""
        motion_times = (times - self.cfg.preparation_duration_s).clamp(min=0) * self.speed
        return self.motion.sample(motion_times, self.clip_ids)

    @property
    def observation_target(self) -> torch.Tensor:
        """Target positions [m] for the next control step, shape [N, 21, 3]."""
        return self.sample_target(self.reference_time + self._env.step_dt)

    @property
    def command(self) -> torch.Tensor:
        """Next-step positions [m], normalized motion phase and playback speed (not a policy observation)."""
        observation_time = self.reference_time + self._env.step_dt
        positions = self.sample_target(observation_time).flatten(1)
        motion_time = (observation_time - self.cfg.preparation_duration_s).clamp(min=0) * self.speed
        phase = (motion_time / self.motion.durations[self.clip_ids]).clamp(0, 1).unsqueeze(1)
        return torch.cat((positions, phase, self.speed.unsqueeze(1)), dim=1)

    def landmarks(self) -> torch.Tensor:
        """Measure landmark positions [m] in rh_palm's local link frame, shape [N, 21, 3]."""
        poses = self.robot.data.body_link_pose_w.torch[:, self.body_ids]
        positions = poses[..., :3] + quat_apply(
            poses[..., 3:].reshape(-1, 4), self.offsets.expand(self.num_envs, -1, -1).reshape(-1, 3)
        ).reshape(self.num_envs, 21, 3)
        palm = poses[:, 0]
        return quat_apply_inverse(
            palm[:, None, 3:].expand(-1, 21, -1).reshape(-1, 4),
            (positions - palm[:, None, :3]).reshape(-1, 3),
        ).reshape(self.num_envs, 21, 3)

    def landmark_velocities(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return palm-relative landmark linear [m/s] and angular [rad/s] velocities in the palm frame."""
        poses = self.robot.data.body_link_pose_w.torch[:, self.body_ids]
        velocities = self.robot.data.body_link_vel_w.torch[:, self.body_ids]
        offsets = quat_apply(
            poses[..., 3:].reshape(-1, 4), self.offsets.expand(self.num_envs, -1, -1).reshape(-1, 3)
        ).reshape(self.num_envs, 21, 3)
        positions = poses[..., :3] + offsets
        linear = velocities[..., :3] + torch.cross(velocities[..., 3:], offsets, dim=-1)
        relative = positions - poses[:, :1, :3]
        linear = linear - velocities[:, :1, :3] - torch.cross(
            velocities[:, :1, 3:].expand_as(relative), relative, dim=-1
        )
        angular = velocities[..., 3:] - velocities[:, :1, 3:]
        palm_rotation = poses[:, :1, 3:].expand(-1, 21, -1).reshape(-1, 4)
        return tuple(
            quat_apply_inverse(palm_rotation, value.reshape(-1, 3)).reshape(self.num_envs, 21, 3)
            for value in (linear, angular)
        )

    def landmark_rotations(self) -> torch.Tensor:
        """Return palm-relative link orientations as two rotation-matrix columns, shape [N, 21, 6]."""
        poses = self.robot.data.body_link_pose_w.torch[:, self.body_ids]
        relative = quat_mul(quat_conjugate(poses[:, :1, 3:]).expand(-1, 21, -1), poses[..., 3:])
        return matrix_from_quat(relative)[..., :2].transpose(-1, -2).flatten(-2)

    def _resample_command(self, env_ids: Sequence[int]):
        self.speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.speed_range)
        self.clip_ids[env_ids] = torch.randint(len(self.motion.words), (len(env_ids),), device=self.device)

    def _update_command(self):
        pass

    def _update_metrics(self):
        error = (self.landmarks() - self.target).norm(dim=-1)
        self.metrics["landmark_error_m"][:] = error[:, 1:].mean(dim=1)
        self.metrics["fingertip_error_m"][:] = error[:, TIPS].mean(dim=1)


@configclass
class HandMotionCommandCfg(CommandTermCfg):
    """Reference dataset and playback settings."""

    class_type = HandMotionCommand
    resampling_time_range = (1e9, 1e9)
    motion_file: str = ""
    """Dataset NPZ of palm-local targets (set by the training and evaluation entry points)."""
    speed_range: tuple[float, float] = (1.0, 1.0)
    """Playback speed range; every episode draws a speed from it (1.0 in all RoboFS runs)."""
    preparation_duration_s: float = 1.0
    """Initial first-pose hold duration [s], before source motion playback."""
