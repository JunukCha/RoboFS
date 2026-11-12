# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Relation-preserving retargeting of generated hands to joint-limited Shadow Hand command trajectories.

Each clip is optimized jointly over time with Adam: the forward-kinematics (FK) landmarks of the robot follow the
generated hand's bone directions and fingertip relations (``robofs.retarget.objective``), joints stay inside
their soft limits through a sigmoid parameterization, and a frame-to-frame penalty keeps consecutive solutions
consistent. Commands use the policy's normalized action space (14 joints + 4 fixed tendons); for FK each tendon
target is split equally between its two driven joints. Self-collision is not modeled.

The output is a dataset NPZ for the tracker: the input dataset with ``poses`` replaced by the robot FK landmarks
[m] and the normalized commands added as ``actions`` [N, T, 18] (``raw_poses``, ``words`` and ``lengths`` are
kept, so the recognizer can map robot motion back to the source image frame).

``--init_file`` starts from given commands (e.g. retargeting-network predictions); with ``--steps 0`` it only
computes their FK landmarks.

Usage:
    python -m robofs.retarget.retarget --dataset_path data/generated/train.npz --output_file data/targets/relation/train.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from robofs.config import section

# Normalized commands of the policy action space: 14 joint positions + 4 fixed-tendon positions.
ACTION_LAYOUT = "14 normalized joint positions (soft limits) + 4 normalized fixed-tendon positions"


def _parse_args() -> argparse.Namespace:
    cfg = section("retarget")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_path", type=Path, required=True, help="Packed generated hands (robofs.generation.pack).")
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=cfg["steps"], help="Adam iterations per batch of clips.")
    parser.add_argument("--init_file", type=Path, help="Initial normalized commands (robofs.retarget.network predict).")
    parser.add_argument("--clips_per_batch", type=int, default=cfg["clips_per_batch"])
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    args = parser.parse_args()
    if args.output_file.exists():
        parser.error(f"Output already exists: {args.output_file}")
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    return args


class _IndexMaps:
    """Map the simulator's joints, tendons and landmark bodies onto the Newton model used for FK."""

    def __init__(self, env):
        term = env.command_manager.get_term("motion")
        self.model = env.sim.physics_manager.get_model()
        starts = self.model.joint_q_start.numpy()
        self.q_indices = {label.rsplit("/", 1)[-1]: int(starts[i]) for i, label in enumerate(self.model.joint_label)}
        labels = {label.rsplit("/", 1)[-1]: i for i, label in enumerate(self.model.body_label)}
        body_ids = [labels[name] for name in term.robot.body_names]
        self.landmark_ids = [body_ids[i] for i in term.body_ids]
        direct_cfg = env.action_manager.get_term("joint_pos").cfg
        self.direct_ids, direct_names = term.robot.find_joints(
            direct_cfg.joint_names, preserve_order=direct_cfg.preserve_order
        )
        tendon_cfg = env.action_manager.get_term("tendon_pos").cfg
        _, tendon_names = term.robot.find_fixed_tendons(tendon_cfg.tendon_names, preserve_order=tendon_cfg.preserve_order)
        self.direct_q = [self.q_indices[name] for name in direct_names]
        # Each fixed tendon drives the J2 and J1 joints of its finger.
        self.tendon_q = [
            [self.q_indices[name.replace("J0", "J2")], self.q_indices[name.replace("J0", "J1")]] for name in tendon_names
        ]


def main() -> None:
    args = _parse_args()
    cfg_retarget = section("retarget")
    sys.argv = [sys.argv[0]]

    import gymnasium as gym
    import newton
    import numpy as np
    import torch

    from isaaclab.app import launch_simulation

    from isaaclab_assets.robots.shadow_hand import TENDON_POSITION_LIMITS
    from isaaclab_tasks.utils import resolve_task_config

    from robofs.retarget.objective import relation_loss, smoothness, source_scale
    from robofs.tracker.task import TASK_NAME, register_task, spawn_hand_without_grasp_marker

    register_task()
    cfg, _ = resolve_task_config(TASK_NAME, "")
    cfg.scene.num_envs = 1
    cfg.scene.robot.spawn.func = spawn_hand_without_grasp_marker
    cfg.commands.motion.motion_file = str(args.dataset_path.resolve())

    with launch_simulation(cfg, args):
        env = gym.make(TASK_NAME, cfg=cfg)
        env.reset()
        raw = env.unwrapped
        device = raw.device
        term = raw.command_manager.get_term("motion")
        maps = _IndexMaps(raw)
        model = maps.model
        joint_type = model.joint_type.numpy()
        parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
        x_parent = torch.tensor(model.joint_X_p.numpy(), device=device, dtype=torch.float32)
        x_child = torch.tensor(model.joint_X_c.numpy(), device=device, dtype=torch.float32)
        axis = torch.tensor(model.joint_axis.numpy(), device=device, dtype=torch.float32)
        q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        q_default = torch.tensor(model.joint_q.numpy(), device=device, dtype=torch.float32)
        revolute = int(newton.JointType.REVOLUTE)
        if not set(joint_type.tolist()) <= {revolute, int(newton.JointType.FIXED)}:
            raise RuntimeError("Retargeting FK supports revolute and fixed joints only.")

        def qmul(a, b):
            ax, ay, az, aw = a.unbind(-1)
            bx, by, bz, bw = b.unbind(-1)
            return torch.stack(
                (
                    aw * bx + ax * bw + ay * bz - az * by,
                    aw * by - ax * bz + ay * bw + az * bx,
                    aw * bz + ax * by - ay * bx + az * bw,
                    aw * bw - ax * bx - ay * by - az * bz,
                ),
                -1,
            )

        def qrot(q, v):
            u, w = q[..., :3], q[..., 3:]
            t = 2 * torch.cross(u, v, dim=-1)
            return v + w * t + torch.cross(u, t, dim=-1)

        def compose(a, b):
            return torch.cat((a[..., :3] + qrot(a[..., 3:], b[..., :3]), qmul(a[..., 3:], b[..., 3:])), -1)

        def invert(a):
            q = torch.cat((-a[..., 3:6], a[..., 6:]), -1)
            return torch.cat((-qrot(q, a[..., :3]), q), -1)

        def forward_kinematics(q):
            """Palm-local landmarks [m], shape [B, 21, 3], for joint coordinates [B, nq] (XYZW quaternions)."""
            batch = q.shape[0]
            bodies = {}
            for j in range(len(joint_type)):
                pose = x_parent[j].expand(batch, 7)
                if parent[j] >= 0:
                    pose = compose(bodies[parent[j]], pose)
                if joint_type[j] == revolute:
                    half = 0.5 * q[:, q_start[j]]
                    rotation = torch.cat((axis[qd_start[j]] * torch.sin(half)[:, None], torch.cos(half)[:, None]), -1)
                    pose = compose(pose, torch.cat((torch.zeros(batch, 3, device=device), rotation), -1))
                bodies[child[j]] = compose(pose, invert(x_child[j]).expand(batch, 7))
            poses = torch.stack([bodies[i] for i in maps.landmark_ids], 1)
            points = poses[..., :3] + qrot(poses[..., 3:], term.offsets.expand(batch, -1, -1))
            palm = poses[:, :1]
            inverse = torch.cat((-palm[..., 3:6], palm[..., 6:]), -1).expand(-1, 21, -1)
            return qrot(inverse, points - palm[..., :3])

        limits = term.robot.data.soft_joint_pos_limits.torch[0, maps.direct_ids]
        low = torch.cat((limits[:, 0], torch.full((4,), float(TENDON_POSITION_LIMITS[0]), device=device)))
        high = torch.cat((limits[:, 1], torch.full((4,), float(TENDON_POSITION_LIMITS[1]), device=device)))
        direct_q = torch.tensor(maps.direct_q, device=device)
        tendon_q = torch.tensor(maps.tendon_q, device=device)

        def joint_coordinates(x):
            """Map unit-box variables [B, 18] to full joint coordinates [B, nq]."""
            values = low + (high - low) * x
            q = q_default.expand(x.shape[0], -1).clone()
            q[:, direct_q] = values[:, :14]
            q[:, tendon_q[:, 0]] = 0.5 * values[:, 14:]
            q[:, tendon_q[:, 1]] = 0.5 * values[:, 14:]
            return q

        # Check the torch FK against the simulator at the reset pose.
        joint_pos = term.robot.data.joint_pos.torch[0]
        q_now = q_default.clone()[None]
        for robot_id, name in enumerate(term.robot.joint_names):
            q_now[0, maps.q_indices[name]] = joint_pos[robot_id]
        fk_error = (forward_kinematics(q_now)[0] - term.landmarks()[0]).norm(dim=-1).max().item()
        if fk_error > 1e-3:
            raise RuntimeError(f"FK mismatch {fk_error:.4f} m against the simulator.")

        motion = term.motion
        scale = source_scale(term.landmarks()[0])
        source_all, lengths_all = motion.points, motion.lengths
        clips, frames = source_all.shape[:2]
        init_actions = None
        if args.init_file is not None:
            with np.load(args.init_file, allow_pickle=False) as data:
                if not np.array_equal(data["words"], np.array(motion.words)):
                    raise ValueError(f"{args.init_file} does not match {args.dataset_path}.")
                init_actions = torch.as_tensor(data["actions"][:, :frames], dtype=torch.float32)
        x_out = torch.zeros(clips, frames, 18, device=device)
        landmark_out = torch.zeros(clips, frames, 21, 3)
        batch_seconds = []
        for start in range(0, clips, args.clips_per_batch):
            batch_start = time.perf_counter()
            stop = min(start + args.clips_per_batch, clips)
            source = source_all[start:stop].to(device)
            lengths = lengths_all[start:stop].to(device)
            batch = stop - start
            valid = torch.arange(frames, device=device)[None] < lengths[:, None]
            if init_actions is None:
                # Start from the mid-range pose; the temporal penalty keeps neighbours consistent.
                logits = torch.zeros(batch, frames, 18, device=device, requires_grad=True)
            else:
                # Normalized commands a = 2x - 1 for joints and tendons alike (the tendon range starts at 0).
                start_x = ((init_actions[start:stop].to(device) + 1) / 2).clamp(1e-4, 1 - 1e-4)
                logits = torch.logit(start_x).requires_grad_(True)
            optimizer = torch.optim.Adam([logits], lr=cfg_retarget["learning_rate"])
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(args.steps, 1))
            for _ in range(args.steps):
                x = torch.sigmoid(logits)
                landmarks = forward_kinematics(joint_coordinates(x.reshape(-1, 18))).reshape(batch, frames, 21, 3)
                tracking = relation_loss(
                    landmarks,
                    source,
                    scale,
                    cfg_retarget["direction_weight"],
                    cfg_retarget["fingertip_weight"],
                    cfg_retarget["rho_m"],
                )
                loss = (tracking * valid).sum() / valid.sum()
                change = smoothness(x)
                loss = loss + cfg_retarget["smooth_weight"] * (change * valid[:, 1:]).sum() / valid[:, 1:].sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                schedule.step()
            with torch.no_grad():
                x = torch.sigmoid(logits)
                landmarks = forward_kinematics(joint_coordinates(x.reshape(-1, 18))).reshape(batch, frames, 21, 3)
                x_out[start:stop] = x
                landmark_out[start:stop] = landmarks.cpu()
            batch_seconds.append(time.perf_counter() - batch_start)
            print(f"step {stop}/{clips} clips retargeted", flush=True)

        values = low + (high - low) * x_out
        tendon_scale = 0.5 * float(TENDON_POSITION_LIMITS[1])
        actions = torch.cat((2 * x_out[..., :14] - 1, values[..., 14:] / tendon_scale - 1), -1).clamp(-1, 1)
        with np.load(args.dataset_path, allow_pickle=False) as data:
            output = {key: data[key] for key in data.files}
        if (
            landmark_out.shape != output["poses"].shape
            or not np.array_equal(np.array(motion.words), output["words"])
            or not np.array_equal(lengths_all.cpu().numpy(), output["lengths"])
        ):
            raise ValueError("Retargeted clips do not match the dataset.")
        metadata = json.loads(str(output["metadata"].item()))
        metadata["target_source"] = {
            "retarget_source_dataset": str(args.dataset_path.resolve()),
            "fps": motion.fps,
            "action_layout": ACTION_LAYOUT,
            "target_mode": "relation",
            "smooth_weight": cfg_retarget["smooth_weight"],
            "steps": args.steps,
            "init_file": str(args.init_file.resolve()) if args.init_file else None,
            "clips_per_batch": args.clips_per_batch,
            "batch_seconds": batch_seconds,
            "limitations": "Equal tendon J1/J2 split FK; no self-collision; joint soft limits enforced.",
            "note": "poses are retargeted robot FK landmarks.",
        }
        output["poses"] = landmark_out.numpy().astype(np.float32)
        output["actions"] = actions.cpu().numpy().astype(np.float32)
        output["metadata"] = np.array(json.dumps(metadata))
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_file, **output)
        print(f"Saved {clips} retargeted clips to {args.output_file}")
        env.close()


if __name__ == "__main__":
    main()
