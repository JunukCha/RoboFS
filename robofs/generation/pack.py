# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Pack a generation folder into one dataset NPZ of palm-aligned hands.

The output holds ``poses`` (palm-local [m], [N, T, 21, 3]), ``raw_poses`` (OpenFS coordinates, used by the
recognizer to map robot motion back to the source image frame), ``lengths``, ``words`` and JSON ``metadata``.
Clips are padded to the longest one by repeating their last frame.

Usage:
    python -m robofs.generation.pack --generation_dir data/generation/test --split test --output_file data/generated/test.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robofs.config import section
from robofs.generation.palm import align_palm


def pack_dataset(manifest_path: Path, output: Path, depth_sign: int, split: str) -> None:
    """Pack every generated clip of a manifest with its palm-aligned targets [m] and provenance."""
    manifest = json.loads(manifest_path.read_text())
    clips, raw_clips, words, provenance = [], [], [], []
    fps = float(manifest["fps"])
    for job in manifest["samples"]:
        if job["status"] not in {"saved", "skipped"}:
            raise ValueError("All generation jobs must be complete before packing.")
        folder = manifest_path.parent / job["path"]
        raw = np.load(folder / "joints_21x3.npy", allow_pickle=False)
        if len(raw) < 2:
            raise ValueError("Each clip needs at least two frames.")
        raw_clips.append(raw)
        clips.append(align_palm(raw, depth_sign))
        words.append(job["text"])
        provenance.append(json.loads((folder / "metadata.json").read_text()))
    if not clips or len(words) != len(set(words)):
        raise ValueError("Expected one complete sequence per distinct word.")
    lengths = np.array([len(clip) for clip in clips], dtype=np.int64)

    def padded(values: list[np.ndarray]) -> np.ndarray:
        return np.stack(
            [np.pad(value, ((0, int(lengths.max()) - len(value)), (0, 0), (0, 0)), mode="edge") for value in values]
        )

    metadata = {
        "fps": fps,
        "units": "meters",
        "ordering": "MediaPipe",
        "coordinate_frame": "Shadow Hand rh_palm body-local",
        "conversion": (
            "Per-frame MCP palm basis; median wrist-middle-MCP scaled to 0.10 m; isotropic image depth assumption"
        ),
        "depth_sign": depth_sign,
        "validated_retargeting": False,
        "evaluation_protocol": "Split membership supplied by caller; compare independent label lists.",
        "generation_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "generation_provenance_sha256": hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
        "provenance": provenance,
        "split": split,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        poses=padded(clips),
        raw_poses=padded(raw_clips),
        lengths=lengths,
        words=np.array(words),
        metadata=np.array(json.dumps(metadata)),
    )
    print(f"Saved {len(words)} clips: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation_dir", type=Path, required=True, help="Output folder of robofs.generation.generate.")
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--split", required=True, help="Split name stored in the metadata, e.g. train or test.")
    parser.add_argument("--depth_sign", type=int, choices=(-1, 1), default=section("generation")["depth_sign"])
    args = parser.parse_args()
    manifest = args.generation_dir.resolve() / "manifest.json"
    if not manifest.is_file():
        parser.error(f"Generation manifest not found: {manifest}")
    if args.output_file.exists():
        parser.error(f"Output already exists: {args.output_file}")
    pack_dataset(manifest, args.output_file.resolve(), args.depth_sign, args.split)


if __name__ == "__main__":
    main()
