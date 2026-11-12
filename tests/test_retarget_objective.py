# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).

"""Relation-preserving retargeting objective."""

import torch

from robofs.retarget.objective import BONES, TIP_PAIRS, relation_loss, smoothness, source_scale


def test_bones_and_tip_pairs():
    assert len(BONES) == 20 and len(TIP_PAIRS) == 7


def test_scaled_copy_has_zero_loss():
    source = torch.randn(2, 5, 21, 3) * 0.03
    source[..., 0, :] = 0
    rest = torch.zeros(21, 3)
    rest[9, 2] = 0.12  # robot palm 20% larger than the 0.10 m source palm
    scale = source_scale(rest)
    assert abs(scale - 1.2) < 1e-6
    loss = relation_loss(scale * source, source, scale)
    assert loss.shape == (2, 5) and loss.abs().max() < 1e-10


def test_weights():
    source = torch.randn(1, 3, 21, 3) * 0.03
    robot = torch.randn(1, 3, 21, 3) * 0.03
    rho = 0.025
    full = relation_loss(robot, source, 1.0)
    directions = relation_loss(robot, source, 1.0, 1.0, 0.0) / rho**2
    tips = relation_loss(robot, source, 1.0, 0.0, 1.0)
    torch.testing.assert_close(full, rho**2 * 3 * directions + 2 * tips)


def test_smoothness():
    x = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]]])
    torch.testing.assert_close(smoothness(x), torch.tensor([[1.0, 4.0]]))
