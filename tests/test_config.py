# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).

"""The settings file holds the final RoboFS settings."""

from robofs.config import ROOT, load_config, path, section


def test_final_settings():
    retarget = section("retarget")
    assert retarget["steps"] == 500
    assert retarget["rho_m"] == 0.025
    assert (retarget["direction_weight"], retarget["fingertip_weight"]) == (3.0, 2.0)
    assert retarget["smooth_weight"] == 1e-3
    tracker = section("tracker")
    assert (tracker["num_envs"], tracker["max_iterations"], tracker["seed"]) == (1024, 1000, 42)
    assert tracker["future_offsets_s"] == [0.1, 0.2, 0.333]
    rewards = tracker["rewards"]
    assert rewards["landmarks"] == {"weight": 3.0, "sigma": 0.025}
    assert rewards["fingertips"] == {"weight": 2.0, "sigma": 0.04}
    assert (rewards["penetration"], rewards["action_rate"]) == (-3.0, -0.005)
    assert section("generation")["frames_per_letter"] == 8


def test_paths_are_inside_the_repository():
    for name in load_config()["paths"]:
        assert path(name).is_relative_to(ROOT)
