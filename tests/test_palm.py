# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).

"""Palm alignment and its inverse used by the recognizer."""

import numpy as np

from robofs.evaluation.recognition import restore_source_coordinates
from robofs.generation.palm import PALM_LENGTH_M, align_palm


def _hand(frames=10, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.normal(scale=0.05, size=(frames, 21, 3)) + 0.5
    raw[:, 9] = raw[:, 0] + [0.0, -0.2, 0.01]  # a clear wrist-to-middle-MCP direction
    raw[:, 5] = raw[:, 9] + [0.05, 0.0, 0.0]
    raw[:, 17] = raw[:, 9] - [0.05, 0.0, 0.0]
    return raw


def test_palm_frame():
    local = align_palm(_hand())
    assert local.shape == (10, 21, 3) and local.dtype == np.float32
    np.testing.assert_allclose(local[:, 0], 0, atol=1e-7)
    # Middle MCP on +z at the palm length; index MCP on +x relative to the little MCP.
    np.testing.assert_allclose(np.median(np.linalg.norm(local[:, 9], axis=-1)), PALM_LENGTH_M, rtol=1e-5)
    np.testing.assert_allclose(local[:, 9, :2], 0, atol=1e-6)
    assert np.all(local[:, 5, 0] > local[:, 17, 0])


def test_restore_inverts_alignment():
    raw = _hand()
    for depth_sign in (1, -1):
        local = align_palm(raw, depth_sign)
        times = np.arange(len(raw)) / 30.0
        projected, source = restore_source_coordinates(local, times, raw, 30.0, depth_sign)
        np.testing.assert_allclose(source, raw, atol=1e-6)
        np.testing.assert_allclose(projected, raw, atol=1e-5)
