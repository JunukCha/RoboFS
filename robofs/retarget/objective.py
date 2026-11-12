# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Relation-preserving retargeting objective (pure PyTorch, no simulator).

The robot hand's forward-kinematics landmarks are matched to the generated hand by two relations instead of
absolute positions: the unit direction of every bone, and the vectors between fingertips (scaled from the source
hand's 0.10 m palm length to the robot hand). Landmarks use the MediaPipe order: 0 wrist, then four per finger
(thumb, index, middle, ring, little) from base to tip.
"""

from __future__ import annotations

import torch

from robofs.generation.palm import PALM_LENGTH_M

# Landmark chains of the five fingers (base joint to tip).
CHAINS = tuple(tuple(range(start, start + 4)) for start in (1, 5, 9, 13, 17))
# Bones: wrist to each finger base, then the three segments of every finger.
BONES = [(0, chain[0]) for chain in CHAINS] + [pair for chain in CHAINS for pair in zip(chain[:-1], chain[1:])]
BONE_PARENTS = [parent for parent, _ in BONES]
BONE_CHILDREN = [child for _, child in BONES]
# Fingertip pairs: thumb to each finger, and neighbouring fingers.
TIP_PAIRS = [(4, 8), (4, 12), (4, 16), (4, 20), (8, 12), (12, 16), (16, 20)]
TIPS_A = [a for a, _ in TIP_PAIRS]
TIPS_B = [b for _, b in TIP_PAIRS]


def source_scale(robot_rest: torch.Tensor) -> float:
    """Ratio of the robot's wrist-to-middle-MCP length [m] to the source hand's 0.10 m."""
    return float((robot_rest[9] - robot_rest[0]).norm()) / PALM_LENGTH_M


def relation_loss(
    landmarks: torch.Tensor,
    source: torch.Tensor,
    scale: float,
    direction_weight: float = 3.0,
    fingertip_weight: float = 2.0,
    rho_m: float = 0.025,
) -> torch.Tensor:
    """Per-frame relation mismatch [m^2] between robot landmarks and the generated hand.

    ``rho^2 * (w_dir * L_dir + w_tip * L_tip / rho^2)``, where ``L_dir`` is the mean squared difference of unit bone
    directions and ``L_tip`` the mean squared difference [m^2] of fingertip-to-fingertip vectors.

    Args:
        landmarks: Robot FK landmarks [m], shape [..., 21, 3].
        source: Palm-aligned generated landmarks [m], shape [..., 21, 3].
        scale: Robot-to-source hand size ratio (:func:`source_scale`).
        direction_weight: Weight of the bone-direction term.
        fingertip_weight: Weight of the fingertip-relation term.
        rho_m: Length scale [m] that puts the direction term on the scale of squared positions.

    Returns:
        Loss per frame, shape [...].
    """
    robot_bones = landmarks[..., BONE_CHILDREN, :] - landmarks[..., BONE_PARENTS, :]
    source_bones = source[..., BONE_CHILDREN, :] - source[..., BONE_PARENTS, :]
    directions = (
        (
            robot_bones / robot_bones.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            - source_bones / source_bones.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        )
        .square()
        .sum(-1)
        .mean(-1)
    )
    vectors = landmarks[..., TIPS_B, :] - landmarks[..., TIPS_A, :]
    source_vectors = scale * (source[..., TIPS_B, :] - source[..., TIPS_A, :])
    tip_terms = (vectors - source_vectors).square().sum(-1).mean(-1) / rho_m**2
    return rho_m**2 * (direction_weight * directions + fingertip_weight * tip_terms)


def smoothness(unit_commands: torch.Tensor) -> torch.Tensor:
    """Squared frame-to-frame change of unit-box commands, shape [B, T - 1] for input [B, T, D]."""
    return (unit_commands[:, 1:] - unit_commands[:, :-1]).square().sum(-1)
