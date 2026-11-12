# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Train RoboFS-Tracker with PPO on a dataset of relation-retargeted targets.

The run goes through Isaac Lab's RSL-RL entry point; logs and checkpoints are written to
``logs/rsl_rl/shadow_hand_landmark_tracking/<date>_<run_name>/`` (``model_1000.pt`` and ``params/``).
Extra arguments are passed on to the entry point, e.g. Hydra overrides such as ``env.rewards.penetration.weight=-3``.

Usage:
    python -m robofs.tracker.train --dataset_path data/targets/relation/train.npz --run_name robofs_tracker
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from robofs.config import section


def main() -> None:
    cfg = section("tracker")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_path", type=Path, required=True, help="Packed relation-retarget targets (train).")
    parser.add_argument("--run_name", default="robofs_tracker")
    parser.add_argument("--num_envs", type=int, default=cfg["num_envs"])
    parser.add_argument("--max_iterations", type=int, default=cfg["max_iterations"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    args, remaining = parser.parse_known_args()
    if not args.dataset_path.is_file():
        parser.error(f"Dataset not found: {args.dataset_path}")

    from isaaclab_rl.entrypoints import run_train_cli

    from robofs.tracker.runner import HandCheckpointRunner
    from robofs.tracker.task import TASK_NAME, register_task

    register_task()
    cli_args = [
        "--rl_library", "rsl_rl",
        "--task", TASK_NAME,
        "--num_envs", str(args.num_envs),
        "--max_iterations", str(args.max_iterations),
        "--seed", str(args.seed),
        "--run_name", args.run_name,
        "--viz", "none",
        *remaining,
        f"env.commands.motion.motion_file={args.dataset_path.resolve()}",
    ]
    with patch("rsl_rl.runners.OnPolicyRunner", HandCheckpointRunner):
        raise SystemExit(run_train_cli(cli_args))


if __name__ == "__main__":
    main()
