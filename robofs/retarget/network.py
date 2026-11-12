# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Retargeting network: a temporal 1D CNN that predicts relation-retargeted commands from generated hands.

It is trained with supervision on the outputs of the optimization (``robofs.retarget.retarget``) and replaces
that optimization when RoboFS-Data is built. Predictions hold ``actions`` [N, T, 18], ``lengths``, ``words`` and
``metadata``; ``robofs.retarget.retarget --init_file <predictions> --steps 0`` turns them into tracker targets
(forward kinematics only).

Usage:
    python -m robofs.retarget.network train --dataset_path data/generated/train.npz \\
        --target_file data/targets/relation/train.npz --val_dataset_path data/generated/validation.npz \\
        --val_target_file data/targets/relation/validation.npz --output_dir checkpoints/retarget_net
    python -m robofs.retarget.network predict --checkpoint checkpoints/retarget_net/model.pt \\
        --dataset_path data/generated/test.npz --output_file predictions.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robofs.config import path, section

# Landmarks are palm-local with a 0.10 m wrist-to-middle-MCP length; this brings them to O(1).
INPUT_SCALE = 10.0
ACTION_LAYOUT = "14 normalized joint positions (soft limits) + 4 normalized fixed-tendon positions"


class RetargetNet(nn.Module):
    """Residual dilated 1D CNN over frames: landmarks [B, T, 21, 3] -> normalized commands [B, T, 18] in [-1, 1]."""

    def __init__(self, channels: int = 256, dilations: tuple[int, ...] = (1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        self.inp = nn.Conv1d(63, channels, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.GELU(),
                nn.Conv1d(channels, channels, 5, padding=2 * d, dilation=d),
                nn.GELU(),
                nn.Conv1d(channels, channels, 1),
            )
            for d in dilations
        )
        self.out = nn.Conv1d(channels, 18, 1)

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        batch, frames = landmarks.shape[:2]
        x = self.inp((INPUT_SCALE * landmarks).reshape(batch, frames, 63).transpose(1, 2))
        for block in self.blocks:
            x = x + block(x)
        return torch.tanh(self.out(x)).transpose(1, 2)


def load_pairs(dataset_path: Path, target_file: Path | None):
    """Return generated poses [N, T, 21, 3], lengths, words and (if given) the target commands [N, T, 18]."""
    with np.load(dataset_path, allow_pickle=False) as data:
        poses, lengths, words = data["poses"].astype(np.float32), data["lengths"], data["words"]
    actions = None
    if target_file is not None:
        with np.load(target_file, allow_pickle=False) as data:
            if not np.array_equal(data["words"], words):
                raise ValueError(f"{target_file} does not match {dataset_path}.")
            actions = data["actions"].astype(np.float32)
    return poses, lengths, words, actions


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the valid frames of each clip."""
    valid = torch.arange(prediction.shape[1], device=prediction.device)[None] < lengths[:, None]
    return ((prediction - target).square().mean(-1) * valid).sum() / valid.sum()


def train(args) -> None:
    device = torch.device(args.device)
    poses, lengths, _, actions = load_pairs(args.dataset_path, args.target_file)
    val = load_pairs(args.val_dataset_path, args.val_target_file)
    poses_t, actions_t, lengths_t = torch.as_tensor(poses), torch.as_tensor(actions), torch.as_tensor(lengths)
    model = RetargetNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps, pct_start=0.05)
    generator = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, start = [], time.time()
    for step in range(1, args.steps + 1):
        index = torch.randint(0, len(poses_t), (args.batch_size,), generator=generator)
        x, y, n = poses_t[index].to(device), actions_t[index].to(device), lengths_t[index].to(device)
        prediction = model(x)
        loss = masked_mse(prediction, y, n)
        # Light temporal smoothness keeps predictions as continuous as the optimized targets.
        valid = torch.arange(x.shape[1], device=device)[None, 1:] < n[:, None]
        change = (prediction[:, 1:] - prediction[:, :-1]).square().mean(-1)
        loss = loss + args.smooth_weight * (change * valid).sum() / valid.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, *val, device)
            minutes = (time.time() - start) / 60
            history.append({"step": step, "train_loss": loss.item(), "val_mse": val_loss, "minutes": minutes})
            print(f"step {step}/{args.steps} loss={loss.item():.5f} " + json.dumps(history[-1]), flush=True)
    torch.save({"model": model.state_dict(), "config": {"input_scale": INPUT_SCALE}}, args.output_dir / "model.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")


@torch.no_grad()
def evaluate(model, poses, lengths, words, actions, device, batch_size: int = 100) -> float:
    """Masked MSE of the predicted commands on a held-out set."""
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(poses), batch_size):
        x = torch.as_tensor(poses[start : start + batch_size], device=device)
        y = torch.as_tensor(actions[start : start + batch_size], device=device)
        n = torch.as_tensor(lengths[start : start + batch_size], device=device)
        valid = torch.arange(x.shape[1], device=device)[None] < n[:, None]
        total += ((model(x) - y).square().mean(-1) * valid).sum().item()
        count += valid.sum().item()
    model.train()
    return total / count


@torch.no_grad()
def predict(args) -> None:
    device = torch.device(args.device)
    model = RetargetNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.eval()
    poses, lengths, words, _ = load_pairs(args.dataset_path, None)
    actions = np.zeros((*poses.shape[:2], 18), dtype=np.float32)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for i in range(0, len(poses), args.batch_size):
        batch = torch.as_tensor(poses[i : i + args.batch_size], device=device)
        actions[i : i + args.batch_size] = model(batch).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - start
    # Hold the last valid command over padded frames, as the optimization output does.
    for clip, length in enumerate(lengths):
        actions[clip, length:] = actions[clip, length - 1]
    metadata = {
        "source_dataset": str(args.dataset_path.resolve()),
        "action_layout": ACTION_LAYOUT,
        "retarget_net_checkpoint": str(args.checkpoint.resolve()),
        "inference_seconds": seconds,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_file, actions=actions, lengths=lengths, words=words, metadata=np.array(json.dumps(metadata))
    )
    print(json.dumps({"clips": len(poses), "inference_seconds": seconds}), flush=True)


def main() -> None:
    cfg = section("retarget_net")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Train on relation-retargeted targets.")
    t.add_argument("--dataset_path", type=Path, required=True, help="Generated hands (train).")
    t.add_argument("--target_file", type=Path, required=True, help="Relation-retargeted targets of the same clips.")
    t.add_argument("--val_dataset_path", type=Path, required=True)
    t.add_argument("--val_target_file", type=Path, required=True)
    t.add_argument("--output_dir", type=Path, required=True)
    t.add_argument("--steps", type=int, default=cfg["steps"])
    t.add_argument("--batch_size", type=int, default=cfg["batch_size"])
    t.add_argument("--lr", type=float, default=cfg["learning_rate"])
    t.add_argument("--smooth_weight", type=float, default=cfg["smooth_weight"])
    t.add_argument("--eval_every", type=int, default=cfg["eval_every"])
    t.add_argument("--seed", type=int, default=cfg["seed"])
    t.add_argument("--device", default=device)
    p = sub.add_parser("predict", help="Predict commands for packed generated hands.")
    p.add_argument("--checkpoint", type=Path, default=path("retarget_net_checkpoint"))
    p.add_argument("--dataset_path", type=Path, required=True)
    p.add_argument("--output_file", type=Path, required=True)
    p.add_argument("--batch_size", type=int, default=cfg["predict_batch_size"])
    p.add_argument("--device", default=device)
    args = parser.parse_args()
    train(args) if args.command == "train" else predict(args)


if __name__ == "__main__":
    main()
