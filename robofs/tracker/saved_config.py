# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Read the environment settings saved with a trained tracker (``params/env.yaml``).

Checkpoints released with the paper were trained before the code was arranged into the ``robofs`` package. Their
settings name functions by the old module paths and list terms that the final configuration disables (zero
weights, unused observation slots, settings fixed to the final values). This module maps the module paths and
removes those entries after checking that they are inactive, so a saved run loads into this release's config
classes with exactly its trained settings. Pure Python: no simulator import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Module paths of the original code base mapped to this package.
MODULE_PATHS = {
    "env.task": "robofs.tracker.task",
    "env.commands": "robofs.tracker.commands",
    "env.tracking_mdp": "robofs.tracker.mdp",
    "env.collision_rewards": "robofs.tracker.penetration",
}
REWARD_TERMS = ("landmarks", "fingertips", "action_rate", "penetration")
OBSERVATION_TERMS = ("body_state", "tracking")


class _ConfigLoader(yaml.FullLoader):
    """Load the slice objects emitted by Isaac Lab without arbitrary object loading."""


_ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node)),
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a settings file written by Isaac Lab (tuples and slices included)."""
    with Path(path).open() as stream:
        return yaml.load(stream, Loader=_ConfigLoader)


def migrate_module_paths(value: Any) -> Any:
    """Rewrite ``module:name`` strings that use the original module paths, recursively."""
    if isinstance(value, dict):
        return {key: migrate_module_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [migrate_module_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(migrate_module_paths(item) for item in value)
    if isinstance(value, str):
        module, separator, name = value.partition(":")
        if separator and module in MODULE_PATHS:
            return f"{MODULE_PATHS[module]}:{name}"
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Saved tracker settings are not supported by this release: {message}")


def prune_inactive_settings(saved_env: dict[str, Any]) -> dict[str, Any]:
    """Remove disabled entries of a saved environment config after checking that they are inactive.

    Removed: reward terms with weight 0, empty observation slots, the tracking input's ``proprioception`` flag
    (must be false), and command settings fixed in this release (``fit_model_lengths`` false,
    ``next_step_target`` true, ``lookahead_s`` unused when ``next_step_target`` is true, ``metadata_file`` null).

    Args:
        saved_env: Parsed ``params/env.yaml``; modified in place.

    Returns:
        The same dictionary.
    """
    rewards = saved_env["rewards"]
    for name in list(rewards):
        if name not in REWARD_TERMS:
            _require(rewards[name] is None or float(rewards[name]["weight"]) == 0.0, f"reward {name} is active")
            del rewards[name]
    _require(all(name in rewards for name in REWARD_TERMS), "a reward term is missing")

    policy = saved_env["observations"]["policy"]
    _require(all(isinstance(policy.get(name), dict) for name in OBSERVATION_TERMS), "body_state/tracking missing")
    for name in list(policy):
        if name in OBSERVATION_TERMS:
            continue
        if policy[name] is None:
            del policy[name]
        else:
            _require(not (isinstance(policy[name], dict) and "func" in policy[name]), f"observation {name} is active")
    params = policy["tracking"].setdefault("params", {})
    _require(not params.pop("proprioception", False), "proprioception observations")

    motion = saved_env["commands"]["motion"]
    _require(not motion.pop("fit_model_lengths", False), "targets refitted to the robot bone lengths (fit_model_lengths)")
    _require(motion.pop("next_step_target", True), "legacy current/future target observations")
    motion.pop("lookahead_s", None)
    _require(motion.pop("metadata_file", None) is None, "separate metadata file")
    _require("preparation_duration_s" in motion, "runs without a preparation hold")
    return saved_env


def load_saved_env(run_dir: Path) -> dict[str, Any]:
    """Return the saved environment settings of a run, ready to merge into ``HandTrackingEnvCfg``.

    Video recording, visualizer and log-directory entries are removed; the saved ``seed`` is kept.
    """
    saved_env = migrate_module_paths(load_yaml(Path(run_dir) / "params/env.yaml"))
    saved_env.pop("video_recorders", None)
    saved_env["sim"].pop("visualizer_cfgs", None)
    saved_env.pop("log_dir", None)
    return prune_inactive_settings(saved_env)


def load_saved_agent(run_dir: Path) -> dict[str, Any]:
    """Return the saved PPO settings of a run."""
    return migrate_module_paths(load_yaml(Path(run_dir) / "params/agent.yaml"))
