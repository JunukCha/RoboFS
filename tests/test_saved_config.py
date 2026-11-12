# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).

"""Saved tracker settings of the original code base load into the release's module layout."""

import pytest
import yaml

from robofs.config import path
from robofs.tracker.saved_config import load_saved_env

SAVED_ENV = """
seed: 42
log_dir: /tmp/run
sim: {dt: 0.005, visualizer_cfgs: []}
video_recorders: []
rewards:
  landmarks: {func: env.tracking_mdp:track_landmarks, params: {sigma: 0.025}, weight: 3.0}
  fingertips: {func: env.tracking_mdp:track_landmarks, params: {sigma: 0.04, fingertips_only: true}, weight: 2.0}
  landmark_velocity: {func: env.tracking_mdp:track_landmark_velocity, params: {sigma: 0.5}, weight: 0.0}
  action_rate: {func: isaaclab.envs.mdp.rewards:action_rate_l2, params: {}, weight: -0.005}
  joint_speed: {func: isaaclab.envs.mdp.rewards:joint_vel_l2, params: {}, weight: 0.0}
  penetration: {func: env.collision_rewards:penetration_penalty, params: {}, weight: -3.0}
observations:
  policy:
    concatenate_terms: true
    previous_actions: null
    body_state: {func: env.tracking_mdp:landmark_state, params: {}}
    tracking:
      func: env.tracking_mdp:landmark_tracking
      params: {future_offsets_s: [0.1, 0.2, 0.333], proprioception: false}
commands:
  motion:
    class_type: env.commands:HandMotionCommand
    motion_file: /data/train.npz
    metadata_file: null
    fit_model_lengths: false
    preparation_duration_s: 1.0
    next_step_target: true
    lookahead_s: 0.15
"""


def _write(tmp_path, text):
    (tmp_path / "params").mkdir()
    (tmp_path / "params/env.yaml").write_text(text)
    return tmp_path


def test_original_settings_load(tmp_path):
    env = load_saved_env(_write(tmp_path, SAVED_ENV))
    assert list(env["rewards"]) == ["landmarks", "fingertips", "action_rate", "penetration"]
    assert env["rewards"]["penetration"]["func"] == "robofs.tracker.penetration:penetration_penalty"
    assert list(env["observations"]["policy"]) == ["concatenate_terms", "body_state", "tracking"]
    assert env["observations"]["policy"]["tracking"]["params"] == {"future_offsets_s": [0.1, 0.2, 0.333]}
    assert env["commands"]["motion"] == {
        "class_type": "robofs.tracker.commands:HandMotionCommand",
        "motion_file": "/data/train.npz",
        "preparation_duration_s": 1.0,
    }
    assert env["seed"] == 42 and "log_dir" not in env and "video_recorders" not in env


def test_active_unsupported_term_is_refused(tmp_path):
    saved = yaml.safe_load(SAVED_ENV)
    saved["rewards"]["joint_speed"]["weight"] = -0.01
    with pytest.raises(ValueError, match="joint_speed"):
        load_saved_env(_write(tmp_path, yaml.safe_dump(saved)))


@pytest.mark.skipif(not (path("tracker_checkpoint").parent / "params/env.yaml").is_file(), reason="not downloaded")
def test_released_tracker_settings():
    env = load_saved_env(path("tracker_checkpoint").parent)
    assert env["seed"] == 42
    assert env["rewards"]["penetration"]["weight"] == -3.0 and env["rewards"]["action_rate"]["weight"] == -0.005
    assert tuple(env["observations"]["policy"]["tracking"]["params"]["future_offsets_s"]) == (0.1, 0.2, 0.333)
