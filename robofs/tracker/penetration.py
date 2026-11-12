# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Self-penetration penalty from the contacts reported by the Newton physics backend."""

from __future__ import annotations

import re

import torch

from isaaclab.utils.math import quat_apply


class BatchedSelfContact:
    """Cache the signed separations [m] of reported hand self-contacts once per control step."""

    def __init__(self, env):
        import warp as wp

        self.wp = wp
        self.manager = env.sim.physics_manager
        model = self.manager.get_model()
        names = set(env.scene["robot"].body_names)
        indices = []
        for label in model.body_label:
            match = re.search(r"/env_(\d+)/Robot/", label)
            indices.append(int(match[1]) if match and label.rsplit("/", 1)[-1] in names else -1)
        self.body_env = torch.tensor(indices, device=env.device)
        self.shape_body = wp.to_torch(model.shape_body).long()
        self.step = -1

    def read(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the environment index and signed separation [m] of every self-contact."""
        if self.step == env.common_step_counter:
            return self.values
        wp = self.wp
        contacts = self.manager.get_contacts()
        count = int(contacts.rigid_contact_count.numpy()[0])
        if count > contacts.rigid_contact_max:
            raise RuntimeError("Self-contact buffer overflow; training rewards would be incomplete.")
        shapes = [wp.to_torch(getattr(contacts, f"rigid_contact_shape{i}"))[:count].long() for i in (0, 1)]
        bodies = [self.shape_body[s.clamp(min=0)] for s in shapes]
        owners = [self.body_env[b.clamp(min=0)] for b in bodies]
        valid = (shapes[0] >= 0) & (shapes[1] >= 0) & (bodies[0] >= 0) & (bodies[1] >= 0)
        valid &= (owners[0] >= 0) & (owners[0] == owners[1]) & (bodies[0] != bodies[1])
        bodies = [b[valid].long() for b in bodies]
        poses = wp.to_torch(self.manager.get_state().body_q)
        points = []
        for side, body in enumerate(bodies):
            point = wp.to_torch(getattr(contacts, f"rigid_contact_point{side}"))[:count][valid]
            points.append(poses[body, :3] + quat_apply(poses[body, 3:], point))
        normal = wp.to_torch(contacts.rigid_contact_normal)[:count][valid]
        margins = sum(wp.to_torch(getattr(contacts, f"rigid_contact_margin{i}"))[:count][valid] for i in (0, 1))
        separation = ((points[1] - points[0]) * normal).sum(-1) - margins
        self.step = env.common_step_counter
        self.values = owners[0][valid], separation
        return self.values


def _contacts(env) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(env, "_fingerspelling_self_contacts"):
        env._fingerspelling_self_contacts = BatchedSelfContact(env)
    return env._fingerspelling_self_contacts.read(env)


def penetration_penalty(env, tolerance_m: float = 0.001, scale_m: float = 0.003) -> torch.Tensor:
    """Sum of squared contact depths beyond a tolerance [m], normalized by a scale [m], per environment."""
    owners, separation = _contacts(env)
    values = ((-separation - tolerance_m).clamp(min=0) / scale_m).square()
    return torch.zeros(env.num_envs, device=env.device).scatter_add_(0, owners, values)
