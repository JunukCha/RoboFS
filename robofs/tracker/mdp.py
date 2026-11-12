# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Observation, reward and termination terms of the landmark-tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from robofs.tracker.motion import TIPS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from robofs.tracker.commands import HandMotionCommand


def motion_term(env: ManagerBasedRLEnv) -> HandMotionCommand:
    """Return the task's reference command."""
    return env.command_manager.get_term("motion")


def landmark_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Robot state: landmark positions [m], 6D rotations, linear [m/s] and angular [rad/s] velocities (315 values)."""
    term = motion_term(env)
    linear, angular = term.landmark_velocities()
    return torch.cat(
        (term.landmarks().flatten(1), term.landmark_rotations().flatten(1), linear.flatten(1), angular.flatten(1)),
        dim=1,
    )


def landmark_tracking(env: ManagerBasedRLEnv, future_offsets_s: tuple[float, ...] = ()) -> torch.Tensor:
    """Tracking inputs: next-step targets [m], position errors [m], velocity errors [m/s] and look-ahead errors [m].

    Args:
        env: Environment.
        future_offsets_s: Look-ahead times [s] after the current reference time; each appends the
            future-target-minus-current-landmark error [m] so the policy can anticipate actuator lag.
    """
    term = motion_term(env)
    target = term.observation_target
    target_velocity = (target - term.target) / env.step_dt
    linear, _ = term.landmark_velocities()
    landmarks = term.landmarks()
    future = [(term.sample_target(term.reference_time + offset) - landmarks).flatten(1) for offset in future_offsets_s]
    return torch.cat(
        (target.flatten(1), (target - landmarks).flatten(1), (target_velocity - linear).flatten(1), *future),
        dim=1,
    )


def track_landmarks(env: ManagerBasedRLEnv, sigma: float, fingertips_only: bool = False) -> torch.Tensor:
    """Gaussian position-tracking reward with tolerance sigma [m], over landmarks 1-20 or the fingertips."""
    term = motion_term(env)
    error = (term.landmarks() - term.target).square().sum(dim=-1)
    ids = TIPS if fingertips_only else list(range(1, 21))
    return torch.exp(-error[:, ids].mean(dim=1) / sigma**2)


def motion_finished(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End the episode with the clip instead of wrapping from the final pose back to the first one."""
    term = motion_term(env)
    return term.reference_time >= term.reference_duration
