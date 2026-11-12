# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Generate hand landmark sequences from text with the OpenFS generator.

The outputs are raw OpenFS XYZ landmarks in its normalized image coordinates, not robot-frame metric
positions; ``robofs.generation.pack`` converts them to palm-local targets. Each word gets its own seed
(the base seed plus a hash of the word), so results do not depend on the word order or the batch size.

Usage:
    python -m robofs.generation.generate --words_file data/word_lists/test_words.txt --output_dir data/generation/test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from robofs.config import path, section


def plan_samples(words: list[str], samples_per_word: int, seed: int) -> list[dict]:
    """Build per-word sample paths and seeds that do not depend on the word-list order.

    Words are stripped, lower-cased and de-duplicated in order of first appearance.
    """
    if samples_per_word <= 0 or not 0 <= seed < 2**32:
        raise ValueError("samples_per_word must be positive and seed must fit uint32.")
    jobs = []
    for word in dict.fromkeys(word.strip().lower() for word in words if word.strip()):
        digest = hashlib.sha256(word.encode()).hexdigest()
        slug = re.sub(r"[^a-z]+", "_", word).strip("_")[:40] or "text"
        for sample in range(samples_per_word):
            sample_seed = (seed + int(digest[:8], 16) + sample) % 2**32
            jobs.append(
                {
                    "text": word,
                    "seed": sample_seed,
                    "path": f"{slug}_{digest[:12]}/sample_{sample:03d}",
                    "status": "planned",
                }
            )
    if not jobs:
        raise ValueError("Provide at least one nonempty word.")
    return jobs


def _parse_args() -> argparse.Namespace:
    cfg = section("generation")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--words_file", type=Path, required=True, help="UTF-8 file with one word or phrase per line.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=path("generator_checkpoint"))
    parser.add_argument("--openfs_dir", type=Path, default=path("openfs_dir"))
    parser.add_argument("--dataset", choices=("chicago", "fsboard"), default=cfg["dataset"])
    parser.add_argument("--samples_per_word", type=int, default=cfg["samples_per_word"])
    parser.add_argument("--frames_per_letter", type=int, default=cfg["frames_per_letter"])
    parser.add_argument("--fps", type=float, default=cfg["fps"], help="Assigned playback rate [Hz].")
    parser.add_argument("--seed", type=int, default=cfg["seed"], help="Base seed; each word adds a hash of its text.")
    parser.add_argument("--batch_size", type=int, default=cfg["batch_size"], help="Words sampled together.")
    parser.add_argument("--resume", action="store_true", help="Skip samples that are already complete.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch_size must be positive.")
    if args.frames_per_letter <= 0 or not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("frames_per_letter and fps must be positive.")
    if not args.checkpoint.is_file():
        parser.error(f"Generator checkpoint not found: {args.checkpoint}")
    return args


def main() -> None:
    """Generate every word of the list and write a manifest of the samples."""
    args = _parse_args()
    jobs = plan_samples(args.words_file.read_text(encoding="utf-8").splitlines(), args.samples_per_word, args.seed)
    source = args.openfs_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    cfg = OmegaConf.load(source / "conf/config_demo_generator_df.yaml")
    chars = OmegaConf.load(source / f"conf/decode/{args.dataset}.yaml").chars
    for job in jobs:
        unsupported = sorted(set(job["text"]) - set(chars[1:]))
        if unsupported:
            raise ValueError(f"Unsupported characters in {job['text']!r}: {unsupported}")
        if len(job["text"]) * args.frames_per_letter >= 1024:
            raise ValueError("Each sequence must stay below OpenFS's 1024-frame padding index.")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "samples": jobs,
        "frames_per_letter": args.frames_per_letter,
        "fps": args.fps,
        "base_seed": args.seed,
        "dataset": args.dataset,
        "checkpoint": str(checkpoint),
    }
    sys.path.insert(0, str(source))
    from lib.utils.dataset import make_frame_labels_wo_zero, normalize_torch, preprocess_word

    vocab = {char: index for index, char in enumerate(chars)}
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = instantiate(cfg.generator, char_size=len(chars) + 1).to(args.device).eval()
    generator.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
    diffusion = instantiate(cfg.diffusion, simple_loss=None).to(args.device).eval()
    # Only deterministic DDIM (eta = 0) draws all noise up front, which keeps batched samples reproducible.
    if diffusion.sampling_name != "ddim" or diffusion.eta != 0:
        raise ValueError("Sampling requires the OpenFS DDIM config with eta = 0.")
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    metadata = {
        "dataset": args.dataset,
        "chars": chars,
        "fps": args.fps,
        "fps_source": "User-assigned playback rate; generator predicts no timing or fps",
        "frames_per_letter": args.frames_per_letter,
        "ordering": "MediaPipe",
        "units": "OpenFS normalized/non-metric coordinates",
        "coordinate_frame": "OpenFS learned image-coordinate representation, not Shadow Hand rh_palm",
        "source_repository": "https://github.com/AIRC-KETI/OpenFS",
        "source_commit": revision,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "sampling": OmegaConf.to_container(cfg.diffusion),
        "note": "Raw XYZ prediction; no palm alignment, handedness correction, metric depth recovery or retargeting.",
    }
    pending = []
    for job in jobs:
        text, seed = job["text"], job["seed"]
        out = root / job["path"]
        expected = {**metadata, "text": text, "seed": seed, "shape": [len(text) * args.frames_per_letter, 21, 3]}
        if out.exists() and any(out.iterdir()):
            if not args.resume:
                raise FileExistsError(f"Sample already exists: {out}. Use --resume or a new output_dir.")
            required = ["metadata.json", "joints_21x3.npy", "joints_21x3_display.npy", "sequence.npz"]
            if not all((out / name).is_file() for name in required):
                raise RuntimeError(f"Incomplete sample: {out}. Use a fresh output_dir to regenerate.")
            if json.loads((out / "metadata.json").read_text()) != expected:
                raise ValueError(f"Resume configuration mismatch: {out}")
            job["status"] = "skipped"
        else:
            pending.append((job, out, expected))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Group similar lengths to limit padding; output paths and seeds do not depend on the order.
    pending.sort(key=lambda item: len(item[0]["text"]))
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        all_labels, noises = [], []
        for job, _, _ in batch:
            _, word_indices = preprocess_word(job["text"], vocab)
            all_labels.append(
                make_frame_labels_wo_zero(word_indices, [args.frames_per_letter] * len(job["text"]), vocab)
            )
            torch.manual_seed(job["seed"])
            np.random.seed(job["seed"])
            # Same draw as OpenFS Diffusion.sampling for a batch of one, so batching keeps each sample's result.
            noises.append(torch.randn([1, len(all_labels[-1]), 21, 3]))
        max_len = max(len(labels) for labels in all_labels)
        # The generator masks frames whose label equals its char_size (len(chars) + 1).
        frame_labels = torch.full((len(batch), max_len), len(chars) + 1, dtype=torch.long)
        sample_poses = torch.zeros(len(batch), max_len, 21, 3)
        for row, (labels, noise) in enumerate(zip(all_labels, noises)):
            frame_labels[row, : len(labels)] = torch.tensor(labels)
            sample_poses[row, : len(labels)] = noise[0]
        frame_labels = frame_labels.to(args.device)
        frame_indices = torch.arange(max_len, device=args.device).expand(len(batch), -1)
        with torch.inference_mode():
            batch_poses = diffusion.ddim_loop(
                generator,
                sample_poses.to(args.device),
                frame_labels,
                frame_indices,
                eta=diffusion.eta,
                ddim_steps=diffusion.ddim_steps,
            )
        for row, (job, out, expected) in enumerate(batch):
            labels = all_labels[row]
            poses = batch_poses[row : row + 1, : len(labels)]
            if list(poses.shape) != [1, *expected["shape"]] or not torch.isfinite(poses).all():
                raise RuntimeError("Invalid generated pose sequence.")
            raw = poses[0].cpu().numpy()
            display = poses.clone()
            display[..., :2] = normalize_torch(display[..., :2], normalize_value=0.5)
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "joints_21x3.npy", raw)
            np.save(out / "joints_21x3_display.npy", display[0].cpu().numpy())
            np.savez_compressed(
                out / "sequence.npz",
                poses=raw,
                frame_labels=np.array(labels),
                frame_indices=np.arange(len(labels)),
                timestamps=np.arange(len(labels)) / args.fps,
            )
            # Written last: it marks the sample as complete for --resume.
            (out / "metadata.json").write_text(json.dumps(expected, indent=2) + "\n")
            job["status"] = "saved"
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"step {min(start + args.batch_size, len(pending))}/{len(pending)} words generated", flush=True)


if __name__ == "__main__":
    main()
