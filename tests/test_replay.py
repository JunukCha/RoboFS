# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).

"""Open-loop replay sends motion frame f at the control step that RoboFS-Data stores it for."""

from types import SimpleNamespace

import numpy as np
import torch

from robofs.evaluation.replay import ReplayPolicy

DT = 1 / 30
PREPARATION_S = 1.0


def _env(words, fps=30.0):
    term = SimpleNamespace(
        motion=SimpleNamespace(words=list(words), fps=fps),
        clip_ids=torch.arange(len(words)),
        speed=torch.ones(len(words)),
        cfg=SimpleNamespace(preparation_duration_s=PREPARATION_S),
        reference_time=torch.zeros(len(words)),
    )
    env = SimpleNamespace(step_dt=DT, device="cpu", command_manager=SimpleNamespace(get_term=lambda name: term))
    return env, term


def _replay_file(tmp_path, words, lengths, frames=12):
    actions = np.zeros((len(words), frames, 18), dtype=np.float32)
    # Encode the clip index and the frame index in the first two commands.
    actions[..., 0] = np.arange(len(words))[:, None] / 10
    actions[..., 1] = np.arange(frames)[None] / 100
    file = tmp_path / "replay.npz"
    np.savez(file, actions=actions, lengths=np.array(lengths), words=np.array(words))
    return file


def test_frame_indexing(tmp_path):
    words = ["ab", "cde"]
    env, term = _env(["cde", "ab"])  # evaluation order differs from the file order
    policy = ReplayPolicy(env, _replay_file(tmp_path, words, [8, 12]))
    first = round(PREPARATION_S / DT) - 1  # robofs.t2a.pack: motion frame 0 is stored at control step `first`
    for step in range(0, first + 20):
        term.reference_time = torch.full((2,), step * DT)
        actions = policy(None)
        expected = np.clip(step - first, 0, None)
        np.testing.assert_allclose(actions[:, 0].numpy(), [0.1, 0.0], atol=1e-6)
        # Frame 0 during the preparation hold, then one frame per step, then the last frame is held.
        np.testing.assert_allclose(actions[:, 1].numpy(), np.minimum(expected, [11, 7]) / 100, atol=1e-6)


def test_commands_are_clipped(tmp_path):
    env, term = _env(["ab"])
    file = tmp_path / "replay.npz"
    np.savez(file, actions=np.full((1, 4, 18), 3.0, np.float32), lengths=np.array([4]), words=np.array(["ab"]))
    assert ReplayPolicy(env, file)(None).max().item() == 1.0
