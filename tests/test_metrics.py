# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).

"""Recognition scoring and trajectory metrics."""

import numpy as np
import pytest

from robofs.config import path
from robofs.evaluation.metrics import compute_physical_metrics, source_motion_trajectory, text_metrics
from robofs.evaluation.recognition import decoding_budget, summarize_recognition


def test_text_metrics():
    assert text_metrics("hello", "hello")["word_accuracy_percent"] == 100.0
    result = text_metrics("hello", "helo")
    assert (result["edit_distance"], result["character_accuracy_percent"], result["word_accuracy_percent"]) == (
        1,
        80.0,
        0.0,
    )


def test_decoding_budget():
    assert decoding_budget(["ab", "cd"]) == 32
    assert decoding_budget(["x" * 40]) == 41


@pytest.mark.skipif(not (path("openfs_dir") / "lib/utils/eval.py").is_file(), reason="OpenFS submodule missing")
def test_htk_letter_accuracy():
    # HTK costs (deletion 7, insertion 7, substitution 10), errors pooled over words before dividing.
    records = [text_metrics("hello", "hello"), text_metrics("world", "word"), text_metrics("abc", "abd")]
    summary = summarize_recognition(records)
    assert summary["words"] == 3
    assert summary["word_accuracy_percent"] == pytest.approx(100 / 3)
    assert summary["pooled_character_accuracy_percent"] == pytest.approx(100 * (1 - 2 / 13))
    assert summary["pooled_levenshtein_accuracy_percent"] == pytest.approx(100 * (1 - 2 / 13))


def test_preparation_frames_are_excluded():
    frames = 40
    hand = np.random.default_rng(0).normal(scale=0.03, size=(frames, 21, 3))
    # Control step k is recorded at time (k + 1) * dt, so the 1 s hold covers the first 30 steps.
    offsets = np.r_[np.zeros(30, int), 1, np.full(frames - 30, 3)]  # step 29: one contact, step 30: two
    data = {
        "actual": hand,
        "target": hand,
        "timestamps": np.arange(1, frames + 1) / 30,
        "preparation_duration_s": np.array(1.0),
        "max_contact_force_n": np.zeros(frames),
        "actions": np.zeros((frames, 18)),
        "reported_contact_depth_m": np.array([0.002, 0.0005, 0.003]),
        "reported_contact_offsets": offsets,
    }
    trimmed = source_motion_trajectory(data)
    assert len(trimmed["timestamps"]) == frames - 30 and trimmed["timestamps"][0] == pytest.approx(1 / 30)
    np.testing.assert_array_equal(trimmed["reported_contact_depth_m"], [0.0005, 0.003])
    metrics = compute_physical_metrics(data)
    assert metrics["penetration"]["frames"] == 1  # only the 3 mm contact after the hold counts
    assert metrics["penetration"]["max_depth_mm"] == pytest.approx(3.0)
    assert metrics["fingertip_mean_error_mm"] == 0.0
