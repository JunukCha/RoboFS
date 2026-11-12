# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Evaluate RoboFS-Tracker (closed loop) or stored commands (open-loop replay) in simulation.

Every word of a target dataset is executed once in parallel environments: a 1 s hold of the first target pose,
then the clip at 30 Hz. The environment settings come from the trained tracker's ``params/env.yaml``. Results,
written to ``--output_dir``:

* ``metrics.json``: pooled landmark and fingertip errors, penetration frame ratio, fingertip jerk and, with
  recognition, letter accuracy (``physical_metrics.recognition.pooled_character_accuracy_percent``) and word
  accuracy (``word_accuracy_percent``), plus per-word reports.
* ``recognition_summary.json``, ``mean_metrics.json``, ``recognition_per_word.csv``.
* with ``--save_trajectories``: ``word_<i>/`` folders with ``trajectory.npz``, ``metrics.json``,
  ``physical_metrics.json``, ``reference.npy`` and ``metadata.json`` (the input of ``robofs.t2a.pack``).

Usage:
    python -m robofs.evaluation.evaluate --method ppo --dataset_path data/targets/relation/test.npz \\
        --output_dir results/test/tracker --save_trajectories
    python -m robofs.evaluation.evaluate --method replay --replay_file predictions.npz \\
        --dataset_path data/targets/relation_net/test.npz --output_dir results/test/t2a --num_envs 256
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robofs.config import path, section

# Policy calls before each batch of words (outputs discarded), kept from the evaluation protocol of the paper.
WARMUP_STEPS = 10


@dataclass
class ClipRecording:
    """Per-control-step rollout records of one clip, including the preparation hold."""

    actual: np.ndarray
    """Measured palm-local landmarks [m], shape [T, 21, 3]."""
    target: np.ndarray
    """Target palm-local landmarks [m], shape [T, 21, 3]."""
    times: np.ndarray
    """Reference times [s], shape [T]."""
    contact_forces: np.ndarray
    """Maximum aggregate link contact force [N], shape [T]."""
    actions: np.ndarray
    """Clipped normalized commands, shape [T, 18]."""
    contact_depths: list
    """Reported contact depths [m] of each step, when the depth sampler is available."""
    joint_positions: np.ndarray
    """Joint positions [rad], shape [T, J]."""
    joint_velocities: np.ndarray
    """Joint velocities [rad/s], shape [T, J]."""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cfg = section("evaluation")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=("ppo", "replay"), default="ppo")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=path("tracker_checkpoint"),
        help="Tracker weights; its folder holds params/{env,agent}.yaml, which also set up replay.",
    )
    parser.add_argument("--replay_file", type=Path, help="Commands to replay (robofs.t2a.model predict).")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Target dataset NPZ of the evaluated words.")
    parser.add_argument(
        "--recognition_source_dataset",
        type=Path,
        help="Dataset whose raw_poses map robot motion back to the image frame (default: --dataset_path).",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=cfg["tracker_num_envs"], help="Parallel environments.")
    parser.add_argument("--save_trajectories", action="store_true", help="Save per-word folders.")
    parser.add_argument("--no_recognition", action="store_true", help="Skip recognizer scoring.")
    parser.add_argument("--recognizer_device", default=cfg["recognizer_device"])
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs < 1:
        parser.error("--num_envs must be positive")
    if args.method == "replay" and (args.replay_file is None or not args.replay_file.is_file()):
        parser.error("--method replay requires an existing --replay_file")
    args.checkpoint = args.checkpoint.resolve()
    if args.method == "ppo" and not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    args.run_dir = args.checkpoint.parent
    if not (args.run_dir / "params/env.yaml").is_file():
        parser.error(f"Saved environment settings not found: {args.run_dir / 'params/env.yaml'}")
    if not args.dataset_path.is_file():
        parser.error(f"Dataset not found: {args.dataset_path}")
    args.dataset_path = args.dataset_path.resolve()
    return args


def _restore_optional_values(config, saved: dict) -> None:
    """Initialize optional fields before the config API's strict type-checked merge."""
    for key, value in saved.items():
        current = config.get(key) if isinstance(config, dict) else getattr(config, key, None)
        if current is None:
            if isinstance(config, dict):
                config[key] = value
            elif hasattr(config, key):
                setattr(config, key, value)
        elif isinstance(value, dict):
            _restore_optional_values(current, value)


def _dataset_jobs(motion, out: Path, save_references: bool) -> list[tuple[str, Path]]:
    """List (word, output folder) per clip; with ``save_references``, write each clip's target and metadata."""
    jobs = []
    for index, word in enumerate(motion.words):
        folder = out / f"word_{index:02d}"
        if save_references:
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "reference.npy", motion.points[index, : int(motion.lengths[index])].cpu().numpy())
            metadata = {**motion.metadata, "text": word}
            # Dataset metadata lists provenance for every clip (~1 MB); keep only this clip's entry.
            provenance = metadata.get("provenance")
            if isinstance(provenance, list) and len(provenance) == len(motion.words):
                metadata["provenance"] = provenance[index]
            (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        jobs.append((word, folder))
    return jobs


def evaluate(argv: list[str] | None = None) -> None:
    """Run the evaluation described in the module docstring."""
    args = _parse_args(argv)
    sys.argv = [sys.argv[0]]

    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import resolve_task_config

    from importlib.metadata import version

    from robofs.evaluation.contacts import ContactDepthSampler
    from robofs.evaluation.metrics import MetricSettings
    from robofs.tracker.motion import HandMotion
    from robofs.tracker.saved_config import load_saved_agent, load_saved_env
    from robofs.tracker.task import TASK_NAME, register_task, spawn_hand_without_grasp_marker

    register_task()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    settings = MetricSettings()
    motion = HandMotion(args.dataset_path, "cpu")
    motion.metadata["source_dataset"] = str(args.dataset_path)
    jobs = _dataset_jobs(motion, out, args.save_trajectories)

    cfg, agent_cfg = resolve_task_config(TASK_NAME, "rsl_rl_cfg_entry_point")
    saved_env = load_saved_env(args.run_dir)
    _restore_optional_values(cfg, saved_env)
    cfg.from_dict(saved_env)
    recognizer = None
    if not args.no_recognition:
        from robofs.evaluation.recognition import TrajectoryRecognizer

        recognizer = TrajectoryRecognizer(args.recognition_source_dataset or args.dataset_path, args.recognizer_device)
    cfg.scene.robot.spawn.func = spawn_hand_without_grasp_marker
    if args.method == "ppo":
        saved_agent = load_saved_agent(args.run_dir)
        _restore_optional_values(agent_cfg, saved_agent)
        agent_cfg.from_dict(saved_agent)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    cfg.scene.num_envs = min(args.num_envs, len(jobs))
    cfg.commands.motion.speed_range = (1.0, 1.0)
    cfg.commands.motion.motion_file = str(args.dataset_path)
    cfg.log_dir = None
    if args.device:
        cfg.sim.device = args.device
    # Record the final physical state of every clip instead of resetting at its end.
    cfg.terminations.motion_end = None
    cfg.episode_length_s = 1e6
    cfg.sim.visualizer_cfgs = []
    cfg.video_recorders = []
    args.visualizer = None
    args.visualizer_explicit = False
    with launch_simulation(cfg, args):
        raw_env = gym.make(TASK_NAME, cfg=cfg)
        for name in ("points", "lengths", "durations"):
            setattr(motion, name, getattr(motion, name).to(raw_env.unwrapped.device))
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        try:
            if args.method == "ppo":
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=cfg.sim.device)
                runner.load(str(args.checkpoint))
                policy = runner.get_inference_policy(device=cfg.sim.device)
            else:
                from robofs.evaluation.replay import ReplayPolicy

                policy = ReplayPolicy(raw_env.unwrapped, args.replay_file)
            reports = _evaluate_batches(
                env, raw_env.unwrapped, policy, motion, jobs, settings, args, recognizer, ContactDepthSampler
            )
        finally:
            env.close()
    _write_dataset_summary(args, out, reports)
    if recognizer is not None:
        recognizer.write_summary(out, [report["physical_metrics"] for report in reports])
    print(f"Saved evaluation to {out.resolve()}")


def _evaluate_batches(env, raw_env, policy, motion, jobs, settings, args, recognizer, sampler_class) -> list[dict]:
    """Evaluate the clips in groups of ``num_envs``, each recorded until its own duration [s]."""
    import torch

    term = raw_env.command_manager.get_term("motion")
    term.motion = motion
    reports = []
    sampler = sampler_class(raw_env.sim.physics_manager, raw_env.scene["robot"].body_names, settings.penetration_tolerance_m)
    if sampler.available and np.any(sampler.body_env[list(sampler.robot_bodies)] < 0):
        raise RuntimeError("Cannot assign reported robot contacts to evaluation environments.")
    dt = raw_env.step_dt
    device = torch.device(raw_env.device)
    for offset in range(0, len(jobs), raw_env.num_envs):
        group = jobs[offset : offset + raw_env.num_envs]
        env.reset()
        ids = torch.arange(offset, offset + len(group), device=device)
        # Unused environments repeat the group's last clip.
        term.clip_ids[:] = ids[-1]
        term.clip_ids[: len(group)] = ids
        raw_env.obs_buf = raw_env.observation_manager.compute(update_history=False)
        with torch.inference_mode():
            for _ in range(WARMUP_STEPS):
                policy(env.get_observations())
        lengths = [math.ceil(float(duration) / dt) for duration in term.reference_duration[: len(group)]]
        buffers = {key: [] for key in ("actual", "target", "times", "forces", "actions", "joints", "velocities")}
        depths = [[] for _ in group]
        with torch.inference_mode():
            for step in range(max(lengths)):
                # Rebuild the current inputs without advancing observation histories or the reference time.
                raw_env.obs_buf = raw_env.observation_manager.compute(update_history=False)
                actions = policy(env.get_observations())
                _, _, done, _ = env.step(actions)
                if done.any():
                    raise RuntimeError("Evaluation reset before completing the reference.")
                forces = raw_env.scene["contacts"].data.net_normal_forces_w_history.torch
                values = dict(
                    actual=term.landmarks(),
                    target=term.target,
                    times=term.reference_time,
                    forces=forces.norm(dim=-1).reshape(raw_env.num_envs, -1).amax(dim=1),
                    actions=actions.clamp(-1, 1),
                    joints=raw_env.scene["robot"].data.joint_pos.torch,
                    velocities=raw_env.scene["robot"].data.joint_vel.torch,
                )
                for key, value in values.items():
                    buffers[key].append(value[: len(group)].clone())
                if sampler.available:
                    sampler.sample()
                    # Group contacts by environment; a stable sort keeps each environment's contact order.
                    order = np.argsort(sampler.last_env_ids, kind="stable")
                    sorted_ids, sorted_depths = sampler.last_env_ids[order], sampler.last_depths[order]
                    env_ids = np.arange(len(lengths))
                    starts = np.searchsorted(sorted_ids, env_ids, side="left")
                    ends = np.searchsorted(sorted_ids, env_ids, side="right")
                    for env_id, length in enumerate(lengths):
                        if step < length:
                            depths[env_id].append(sorted_depths[starts[env_id] : ends[env_id]].copy())
        arrays = {key: torch.stack(value).cpu().numpy() for key, value in buffers.items()}
        for env_id, ((word, folder), length) in enumerate(zip(group, lengths)):
            clip = {key: value[:length, env_id] for key, value in arrays.items()}
            recording = ClipRecording(
                actual=clip["actual"],
                target=clip["target"],
                times=clip["times"],
                contact_forces=clip["forces"],
                actions=clip["actions"],
                contact_depths=depths[env_id],
                joint_positions=clip["joints"],
                joint_velocities=clip["velocities"],
            )
            metadata = {**term.motion.metadata, "text": word}
            report = _save_clip(raw_env, args, folder, settings, sampler, recording, recognizer, metadata)
            reports.append({"word": word, **report})
        print(f"step {min(offset + len(group), len(jobs))}/{len(jobs)} words evaluated", flush=True)
    return reports


def _save_clip(raw_env, args, out: Path, settings, sampler, recording: ClipRecording, recognizer, metadata) -> dict:
    """Score one clip (preparation frames excluded) and optionally save its trajectory."""
    from robofs.evaluation.metrics import TIPS, compute_physical_metrics, source_motion_trajectory

    times = np.asarray(recording.times)
    steps = len(times)
    term = raw_env.command_manager.get_term("motion")
    preparation = term.cfg.preparation_duration_s
    scoring_mask = times > preparation + 1e-6 if preparation > 0 else np.ones(steps, dtype=bool)
    if scoring_mask.sum() < 2:
        raise ValueError("At least two source-motion frames must remain after preparation.")
    actual, target = np.asarray(recording.actual), np.asarray(recording.target)
    errors = np.linalg.norm(actual - target, axis=-1)
    if not np.isfinite(errors).all():
        raise RuntimeError("Non-finite evaluation trajectory.")
    scored_errors = errors[scoring_mask]
    tips = scored_errors[:, TIPS]
    actions = np.asarray(recording.actions)
    summary = {
        "checkpoint": str(args.checkpoint) if args.method == "ppo" else None,
        "method": args.method,
        "reference": str(out / "reference.npy") if args.save_trajectories else str(args.dataset_path),
        "deterministic_actions": True,
        "speed": 1.0,
        "frames": int(scoring_mask.sum()),
        "recorded_frames": steps,
        "excluded_preparation_frames": int((~scoring_mask).sum()),
        "preparation_duration_s": preparation,
        "dt_s": raw_env.step_dt,
        "landmark_mean_error_m": float(scored_errors[:, 1:].mean()),
        "landmark_max_error_m": float(scored_errors[:, 1:].max()),
        "fingertip_mean_error_m": float(tips.mean()),
        "fingertip_max_error_m": float(tips.max()),
        "fingertip_mean_error_by_finger_m": dict(
            zip(("thumb", "index", "middle", "ring", "little"), tips.mean(axis=0).tolist())
        ),
        "max_aggregate_link_contact_force_n": float(np.asarray(recording.contact_forces)[scoring_mask].max()),
        "mean_squared_action_change": float((np.diff(actions[scoring_mask], axis=0) ** 2).mean()),
    }
    trajectory = dict(
        actual=actual,
        target=target,
        timestamps=times,
        preparation_duration_s=np.array(preparation),
        errors=errors,
        actions=actions,
        max_contact_force_n=np.asarray(recording.contact_forces),
        joint_positions_rad=np.asarray(recording.joint_positions),
        joint_velocities_rad_s=np.asarray(recording.joint_velocities),
        joint_names=np.array(raw_env.scene["robot"].joint_names),
    )
    if sampler.available:
        contact_depths = recording.contact_depths
        trajectory.update(
            reported_contact_depth_m=np.concatenate(contact_depths),
            reported_contact_offsets=np.r_[0, np.cumsum([len(d) for d in contact_depths])],
        )
    scoring_trajectory = source_motion_trajectory(trajectory)
    summary["physical_metrics"] = compute_physical_metrics(scoring_trajectory, settings)
    if not sampler.available:
        summary["physical_metrics"]["penetration"]["reason"] = sampler.reason
    if recognizer is not None:
        summary["physical_metrics"]["recognition"] = recognizer.score(metadata["text"], scoring_trajectory, metadata)
    if args.save_trajectories:
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "trajectory.npz", **trajectory)
        (out / "physical_metrics.json").write_text(
            json.dumps(summary["physical_metrics"], indent=2, allow_nan=False) + "\n"
        )
        (out / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def _write_dataset_summary(args, out: Path, reports: list[dict]) -> None:
    """Write pooled metrics over all words to ``metrics.json``."""
    frames = sum(report["frames"] for report in reports)
    summary = {
        "checkpoint": str(args.checkpoint) if args.method == "ppo" else None,
        "run_dir": str(args.run_dir),
        "method": args.method,
        "replay_file": str(args.replay_file.resolve()) if args.method == "replay" else None,
        "reference_dataset": str(args.dataset_path),
        "words": [report["word"] for report in reports],
        "per_word": reports,
        "word_macro_landmark_mean_error_m": float(np.mean([r["landmark_mean_error_m"] for r in reports])),
        "word_macro_fingertip_mean_error_m": float(np.mean([r["fingertip_mean_error_m"] for r in reports])),
        "pooled_landmark_mean_error_m": sum(r["landmark_mean_error_m"] * r["frames"] for r in reports) / frames,
        "pooled_fingertip_mean_error_m": sum(r["fingertip_mean_error_m"] * r["frames"] for r in reports) / frames,
    }
    physical = [r["physical_metrics"] for r in reports]
    contact_frames = sum(p["contact_frames"] for p in physical)
    jerk_samples = sum(p["jerk_samples"] for p in physical)
    contact_jerk_samples = sum(p["contact_jerk_samples"] for p in physical)
    summary["physical_metrics"] = {
        "settings": physical[0]["settings"],
        "fingertip_mean_error_mm": summary["pooled_fingertip_mean_error_m"] * 1000,
        "fingertip_max_error_mm": max(p["fingertip_max_error_mm"] for p in physical),
        "contact_fingertip_mean_error_mm": sum(
            (p["contact_fingertip_mean_error_mm"] or 0) * p["contact_frames"] for p in physical
        )
        / contact_frames
        if contact_frames
        else None,
        "excessive_contact_frame_ratio_percent": 100 * sum(p["excessive_contact_frames"] for p in physical) / frames,
        "contact_stall_frame_ratio_percent": 100
        * sum(p["contact_stall_frames"] for p in physical)
        / sum(p["stall_evaluable_frames"] for p in physical),
        "fingertip_jerk_rms_m_s3": float(
            np.sqrt(sum((p["fingertip_jerk_rms_m_s3"] or 0) ** 2 * p["jerk_samples"] for p in physical) / jerk_samples)
        )
        if jerk_samples
        else None,
        "contact_fingertip_jerk_rms_m_s3": float(
            np.sqrt(
                sum((p["contact_fingertip_jerk_rms_m_s3"] or 0) ** 2 * p["contact_jerk_samples"] for p in physical)
                / contact_jerk_samples
            )
        )
        if contact_jerk_samples
        else None,
        "recognition": {"available": False, "reason": "No robot recognizer predictions."},
    }
    if all(p["penetration"]["available"] for p in physical):
        count = sum(p["penetration"]["penetrating_contact_samples"] for p in physical)
        summary["physical_metrics"]["penetration"] = {
            "available": True,
            "frame_ratio_percent": 100 * sum(p["penetration"]["frames"] for p in physical) / frames,
            "max_depth_mm": max(p["penetration"]["max_depth_mm"] for p in physical),
            "mean_contact_depth_mm": sum(
                p["penetration"]["mean_contact_depth_mm"] * p["penetration"]["penetrating_contact_samples"]
                for p in physical
            )
            / count
            if count
            else 0.0,
        }
    else:
        summary["physical_metrics"]["penetration"] = {
            "available": False,
            "reason": "Reported contact geometry is unavailable for some evaluations.",
        }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(reports)}-word evaluation: {out / 'metrics.json'}")


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
