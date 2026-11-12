# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Read the hand self-contact depths reported by the Newton physics backend, without changing the physics."""

from __future__ import annotations

import re

import numpy as np


def surface_separation(points0, points1, normals, poses0, poses1, margins0, margins1) -> np.ndarray:
    """Signed surface separation [m] from body-local contact points and XYZW body poses (negative: penetration)."""

    def world(points, poses):
        q = poses[:, 3:7]
        cross = 2 * np.cross(q[:, :3], points)
        return points + q[:, 3:4] * cross + np.cross(q[:, :3], cross) + poses[:, :3]

    return np.sum((world(points1, poses1) - world(points0, poses0)) * normals, axis=-1) - margins0 - margins1


class ContactDepthSampler:
    """Sample the reported self-contact depths [m] of every robot hand in the scene, per environment.

    Args:
        manager: Physics manager of the simulation (Newton backend).
        robot_body_names: Body names of the robot; contacts between other bodies are ignored.
        tolerance_m: Depth [m] above which a contact counts as penetrating.
    """

    def __init__(self, manager, robot_body_names: list[str], tolerance_m: float):
        self.manager = manager
        self.tolerance_m = tolerance_m
        self.available = all(hasattr(manager, name) for name in ("get_contacts", "get_model", "get_state"))
        self.reason = "Active physics backend does not expose Newton contact geometry."
        if not self.available:
            return
        if manager.get_contacts() is None:
            self.available = False
            self.reason = "No Newton contact buffer is available."
            return
        model = manager.get_model()
        self.shape_body = model.shape_body.numpy()
        self.robot_bodies = {i for i, label in enumerate(model.body_label) if label.rsplit("/", 1)[-1] in robot_body_names}
        owners = []
        for label in model.body_label:
            match = re.search(r"/env_(\d+)/Robot/", label)
            owners.append(int(match[1]) if match else -1)
        self.body_env = np.asarray(owners)
        self.last_depths = np.empty(0)
        self.last_env_ids = np.empty(0, dtype=int)

    def sample(self) -> None:
        """Read the current contacts into ``last_depths`` [m] and their environment ids ``last_env_ids``."""
        contacts = self.manager.get_contacts()
        self.last_depths = np.empty(0)
        self.last_env_ids = np.empty(0, dtype=int)
        n = int(contacts.rigid_contact_count.numpy()[0])
        if n > contacts.rigid_contact_max:
            raise RuntimeError("Contact buffer overflow; penetration metrics would be incomplete.")
        if n == 0:
            return
        shapes0 = contacts.rigid_contact_shape0.numpy()[:n]
        shapes1 = contacts.rigid_contact_shape1.numpy()[:n]
        rows = np.flatnonzero((shapes0 >= 0) & (shapes1 >= 0))
        bodies0, bodies1 = self.shape_body[shapes0[rows]], self.shape_body[shapes1[rows]]
        keep = np.array(
            [a in self.robot_bodies and b in self.robot_bodies and a != b for a, b in zip(bodies0, bodies1)],
            dtype=bool,
        )
        keep &= self.body_env[bodies0] == self.body_env[bodies1]
        rows, bodies0, bodies1 = rows[keep], bodies0[keep], bodies1[keep]
        if not len(rows):
            return
        poses = self.manager.get_state().body_q.numpy()
        separation = surface_separation(
            contacts.rigid_contact_point0.numpy()[rows],
            contacts.rigid_contact_point1.numpy()[rows],
            contacts.rigid_contact_normal.numpy()[rows],
            poses[bodies0],
            poses[bodies1],
            contacts.rigid_contact_margin0.numpy()[rows],
            contacts.rigid_contact_margin1.numpy()[rows],
        )
        self.last_depths = np.maximum(-separation, 0)
        self.last_env_ids = self.body_env[bodies0]
