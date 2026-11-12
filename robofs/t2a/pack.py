# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Pack per-word tracker rollouts into one RoboFS-Data shard of (text, robot action) pairs.

The input is the output folder of ``robofs.evaluation.evaluate --save_trajectories`` (``word_*`` folders and
``recognition_per_word.csv``). Only the motion phase is kept: the control step whose target is motion frame 0 and
every later step of the clip, so frame ``f`` of ``actions`` is the command that open-loop replay sends for motion
frame ``f``. The preparation hold before it (1 s on the first pose) is dropped.

Shard contents (N words, T frames at 30 Hz, padded by holding the last frame):
    words [N], lengths [N], actions [N, T, 18], joint_positions_rad [N, T, 24], landmarks [N, T, 21, 3],
    character_accuracy_percent [N], edit_distance [N], penetration_frame_percent [N], fingertip_jerk_rms_m_s3 [N],
    metadata (JSON string).

Usage:
    python -m robofs.t2a.pack --rollout_dir results/robofs_data/part_00/rollout --output_file data/robofs_data/part_00.npz
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ACTION_LAYOUT = "14 normalized joint positions (soft limits) + 4 normalized fixed-tendon positions"


def load_rollout(rollout_dir: Path) -> dict[str, np.ndarray]:
    """Read each ``word_*`` folder of a rollout into padded arrays with per-word quality scores."""
    with (rollout_dir / "recognition_per_word.csv").open() as file:
        recognition = {row["expected_text"]: row for row in csv.DictReader(file) if row["stream"] == "robot"}
    words, actions, joints, landmarks, quality = [], [], [], [], []
    folders = sorted(rollout_dir.glob("word_*"), key=lambda folder: int(folder.name.split("_")[1]))
    target_lengths = {}
    for folder in folders:
        metadata = json.loads((folder / "metadata.json").read_text())
        text = metadata["text"]
        if metadata["source_dataset"] not in target_lengths:
            with np.load(metadata["source_dataset"], allow_pickle=False) as data:
                target_lengths[metadata["source_dataset"]] = dict(zip(data["words"].tolist(), data["lengths"]))
        length = int(target_lengths[metadata["source_dataset"]][text])
        physical = json.loads((folder / "physical_metrics.json").read_text())
        with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
            dt = float(data["timestamps"][0])
            # Step k (1-based) is judged at time k * dt, so motion frame 0 is step round(preparation / dt).
            # Episodes may record one step past the clip end, so the source clip length bounds the slice.
            first = round(float(data["preparation_duration_s"]) / dt) - 1
            if len(data["actions"]) < first + length:
                raise ValueError(f"{folder} is shorter than its {length}-frame clip.")
            words.append(text)
            actions.append(data["actions"][first : first + length])
            joints.append(data["joint_positions_rad"][first : first + length])
            landmarks.append(data["actual"][first : first + length])
        row = recognition[text]
        quality.append(
            (
                float(row["character_accuracy_percent"]),
                float(row["edit_distance"]),
                float(physical["penetration"]["frame_ratio_percent"]),
                float(physical["fingertip_jerk_rms_m_s3"]),
            )
        )
    if not words:
        raise ValueError(f"No word_* folders in {rollout_dir}.")
    lengths = np.array([len(a) for a in actions])
    frames = lengths.max()

    def pad(items: list[np.ndarray]) -> np.ndarray:
        # Hold the last frame over padding, as the target files do.
        return np.stack([np.concatenate([x, np.repeat(x[-1:], frames - len(x), axis=0)]) for x in items])

    quality = np.array(quality, dtype=np.float32)
    return {
        "words": np.array(words),
        "lengths": lengths,
        "actions": pad(actions).astype(np.float32),
        "joint_positions_rad": pad(joints).astype(np.float32),
        "landmarks": pad(landmarks).astype(np.float32),
        "character_accuracy_percent": quality[:, 0],
        "edit_distance": quality[:, 1],
        "penetration_frame_percent": quality[:, 2],
        "fingertip_jerk_rms_m_s3": quality[:, 3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollout_dir", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    args = parser.parse_args()
    if args.output_file.exists():
        parser.error(f"Output already exists: {args.output_file}")
    shard = load_rollout(args.rollout_dir)
    metadata = {
        "source_rollout": str(args.rollout_dir.resolve()),
        "fps": 30.0,
        "action_layout": ACTION_LAYOUT,
        "landmarks": "Simulated robot landmarks [m], Shadow Hand rh_palm body-local, MediaPipe ordering",
        "quality": "Recognizer character accuracy and edit distance on the robot stream, penetration frames, jerk",
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_file, **shard, metadata=np.array(json.dumps(metadata)))
    print(json.dumps({"words": len(shard["words"]), "max_frames": int(shard["lengths"].max())}))


if __name__ == "__main__":
    main()
