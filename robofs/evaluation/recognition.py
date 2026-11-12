# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Score robot motion with the OpenFS fingerspelling recognizer.

Robot landmarks are palm-local [m]. They are mapped back into the generated hand's non-metric image frame (the
inverse of the palm alignment of ``robofs.generation.palm``, using the generated hand's own palm orientation and
scale over time), normalized per frame as OpenFS does, and decoded greedily. Letter accuracy follows OpenFS
``compute_acc``: errors from its HTK-cost alignment (deletion 7, insertion 7, substitution 10) are summed over all
words before dividing by the number of reference characters. Word accuracy is the exact-match rate.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from robofs.config import path
from robofs.evaluation.metrics import text_metrics
from robofs.generation.palm import PALM_LENGTH_M

STREAMS = ("robot", "target", "raw_generated", "raw_native")
"""Scored sequences: simulated robot, tracker target, generated hand resampled to the robot timestamps, and the
generated hand at its own frames."""


def restore_source_coordinates(
    local: np.ndarray, times: np.ndarray, raw: np.ndarray, fps: float, depth_sign: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map palm-local landmarks [m] back into the generated hand's non-metric image frame.

    Args:
        local: Robot or target positions [m], shape [T, 21, 3].
        times: Motion timestamps [s] (preparation removed), shape [T].
        raw: Generated OpenFS coordinates, shape [L, 21, 3], without padding.
        fps: Frame rate [Hz] of ``raw``.
        depth_sign: Depth reflection used by the palm alignment, +1 or -1.

    Returns:
        Projected positions and ``raw`` linearly resampled at ``times``, both in OpenFS coordinates.
    """
    local, times, raw = map(np.asarray, (local, times, raw))
    if (
        raw.ndim != 3
        or raw.shape[1:] != (21, 3)
        or len(raw) < 2
        or local.shape != (len(times), 21, 3)
        or times.ndim != 1
        or not all(np.isfinite(v).all() for v in (local, times, raw))
        or not np.isfinite(fps)
        or fps <= 0
        or depth_sign not in (-1, 1)
        or np.any(times < 0)
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Invalid source projection inputs.")
    sample_times = np.arange(len(raw)) / fps
    source = np.stack(
        [np.interp(times, sample_times, raw[:, j, d]) for j in range(21) for d in range(3)], axis=-1
    ).reshape(-1, 21, 3)
    centered = source - source[:, :1]
    centered[..., 2] *= depth_sign
    z = centered[:, 9].copy()
    z_norm = np.linalg.norm(z, axis=-1)
    if np.any(z_norm < 1e-7):
        raise ValueError("Degenerate source palm direction.")
    z /= z_norm[:, None]
    x = centered[:, 5] - centered[:, 17]
    x -= (x * z).sum(axis=-1, keepdims=True) * z
    x_norm = np.linalg.norm(x, axis=-1)
    if np.any(x_norm < 1e-7):
        raise ValueError("Degenerate source palm basis.")
    x /= x_norm[:, None]
    basis = np.stack((x, np.cross(z, x), z), axis=-1)
    scale = PALM_LENGTH_M / np.median(np.linalg.norm(raw[:, 9] - raw[:, 0], axis=-1))
    projected = np.einsum("tjk,tik->tji", local, basis) / scale
    projected[..., 2] *= depth_sign
    projected += source[:, :1]
    return projected.astype(np.float32), source.astype(np.float32)


def decoding_budget(texts) -> int:
    """Decoding budget in tokens including EOS, as in OpenFS's generator evaluation: max(32, longest label + 1)."""
    return max(32, max((len(str(text)) for text in texts), default=0) + 1)


@functools.cache
def _openfs_compute_acc():
    spec = importlib.util.spec_from_file_location("openfs_eval", path("openfs_dir") / "lib/utils/eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_acc


def summarize_recognition(records: list[dict]) -> dict:
    """Pooled letter accuracy (OpenFS ``compute_acc``) and word accuracy [%] over all words."""
    if not records:
        raise ValueError("Recognition records must be nonempty.")
    characters = sum(len(r["expected_text"]) for r in records)
    openfs_accuracy = _openfs_compute_acc()(
        [r["recognized_text"] for r in records], [r["expected_text"] for r in records]
    )[0]
    return {
        "words": len(records),
        "word_accuracy_percent": float(np.mean([r["word_accuracy_percent"] for r in records])),
        "pooled_character_accuracy_percent": float(openfs_accuracy),
        # Same pooling with unit-cost minimum edit distance; at most a few tenths above the OpenFS value.
        "pooled_levenshtein_accuracy_percent": 100 * (1 - sum(r["edit_distance"] for r in records) / characters),
        "reference_characters": characters,
        "aggregation": "Errors and characters summed over all words (OpenFS compute_acc).",
    }


class TrajectoryRecognizer:
    """Load the OpenFS recognizer once and score palm-local robot trajectories in memory.

    Args:
        source_dataset: Dataset NPZ whose ``raw_poses`` hold the generated hands of the evaluated words.
        device: Torch device of the recognizer.
        checkpoint: Recognizer weights; its Hydra config is read from ``.hydra/config.yaml`` beside them.
        cpu_threads: Torch CPU threads.
    """

    def __init__(self, source_dataset: Path, device: str = "cuda", checkpoint: Path | None = None, cpu_threads: int = 4):
        torch.set_num_threads(cpu_threads)
        source = path("openfs_dir").resolve()
        checkpoint = Path(checkpoint or path("recognizer_checkpoint")).resolve()
        config = checkpoint.parent / ".hydra/config.yaml"
        cfg = OmegaConf.load(config)
        if cfg.recognizer.input_dim != 42:
            raise ValueError("This source-image projection supports 2D recognizers (input_dim=42) only.")
        sys.path.insert(0, str(source))
        from lib.utils.dataset import normalize_torch

        chars = str(cfg.decode.chars)
        recognizer_cfg = OmegaConf.to_container(cfg.recognizer, resolve=True)
        recognizer_cfg.pop("ckpt_path", None)
        self.device = torch.device(device)
        model = instantiate(recognizer_cfg, char_size=len(chars) + 1).to(self.device)
        model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        model.eval()

        source_dataset = Path(source_dataset)
        with np.load(source_dataset, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            poses, lengths, words = data["raw_poses"], data["lengths"], data["words"]
            raw = {str(word): poses[i, : int(lengths[i])].copy() for i, word in enumerate(words)}
        del poses
        if metadata.get("coordinate_frame") != "Shadow Hand rh_palm body-local":
            raise ValueError("Source dataset must use the palm-local conversion of robofs.generation.pack.")
        self.max_len = decoding_budget(raw)
        self.provenance = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "config": str(config),
            "source_dataset": str(source_dataset.resolve()),
            "device": str(self.device),
            "max_len": self.max_len,
            "hand_segment": 0,
            "projection": "Inverse source-driven MCP palm basis and scale, then OpenFS per-frame XY normalize to +/-0.5.",
            "limitations": (
                "Uses generated source orientation, not measured wrist orientation; recognizer domain shift remains."
            ),
        }
        self.model, self.cfg, self.chars, self.normalize = model, cfg, chars, normalize_torch
        self.raw, self.metadata = raw, metadata
        self.records = []

    def score(self, word: str, trajectory: dict, clip_metadata: dict) -> dict:
        """Score one trajectory (preparation removed) with palm-local ``actual``/``target`` [m] and ``timestamps`` [s]."""
        return self.score_batch([(word, trajectory, clip_metadata)])[0]

    def score_batch(self, items: list[tuple[str, dict, dict]], batch_size: int = 64) -> list[dict]:
        """Score trajectories, batching recognizer calls across words and streams.

        Padding follows OpenFS's recognition collate function, so predictions match one-by-one scoring.

        Returns:
            One report per item, in input order; reports are also appended to ``records``.
        """
        model, cfg, chars, device = self.model, self.cfg, self.chars, self.device
        raw, metadata = self.raw, self.metadata
        fps, depth_sign = float(metadata["fps"]), int(metadata["depth_sign"])
        reports, pending = [], []
        for word, trajectory, clip_metadata in items:
            for key in ("generation_manifest_sha256", "depth_sign"):
                if key in clip_metadata and clip_metadata[key] != metadata.get(key):
                    raise ValueError(f"Source dataset does not match evaluation provenance: {key}")
            if word not in raw or any(char not in chars[1:] for char in word):
                raise ValueError(f"Missing source or unsupported label: {word!r}")
            actual, original = restore_source_coordinates(
                trajectory["actual"], trajectory["timestamps"], raw[word], fps, depth_sign
            )
            target, _ = restore_source_coordinates(
                trajectory["target"], trajectory["timestamps"], raw[word], fps, depth_sign
            )
            report = {"available": True, "provenance": self.provenance, "streams": {}}
            reports.append(report)
            for name, poses in zip(STREAMS, (actual, target, original, raw[word])):
                if len(poses) >= cfg.recognizer.max_len:
                    raise ValueError("Sequence exceeds recognizer frame-position capacity; do not silently truncate.")
                inputs = self.normalize(torch.tensor(poses[None, ..., :2], device=device), normalize_value=0.5)
                pending.append((report, name, word, inputs[0]))
        # Similar lengths share a batch to limit padding.
        pending.sort(key=lambda item: len(item[3]))
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            lengths = [len(item[3]) for item in chunk]
            batch, length = len(chunk), max(lengths)
            # OpenFS paired_collate_recog_fn padding: poses -2, hand segment 32, frame index 1024.
            poses = torch.full((batch, length, *chunk[0][3].shape[1:]), -2.0, device=device)
            hand_segment = torch.full((batch, length), 32, dtype=torch.long, device=device)
            frame_idx = torch.full((batch, length), 1024, dtype=torch.long, device=device)
            for row, (item, n) in enumerate(zip(chunk, lengths)):
                poses[row, :n] = item[3]
                hand_segment[row, :n] = 0
                frame_idx[row, :n] = torch.arange(n, device=device)
            with torch.inference_mode():
                output_ids = model.generate(
                    poses, hand_segment, frame_idx, max_len=self.max_len, bos_token_id=len(chars), eos_token_id=0
                )["output_ids"][:, 1:].cpu().tolist()
            for (report, name, word, _), tokens, n in zip(chunk, output_ids, lengths):
                prediction, invalid = "", []
                for token in tokens:
                    if token == 0:
                        break
                    if 0 < token < len(chars):
                        prediction += chars[token]
                    else:
                        prediction += "�"
                        invalid.append(token)
                report["streams"][name] = {
                    **text_metrics(word, prediction),
                    "eos_reached": 0 in tokens,
                    "invalid_token_ids": invalid,
                    "frames": n,
                }
        for report in reports:
            report.update(report["streams"]["robot"])
        self.records.extend(reports)
        return reports

    def write_summary(self, output: Path, physical_reports: list[dict]) -> None:
        """Write ``recognition_summary.json``, ``mean_metrics.json``, ``recognition_per_word.csv``, update ``metrics.json``."""
        records, provenance = self.records, self.provenance
        summary = {
            "provenance": provenance,
            "recognition": {name: summarize_recognition([r["streams"][name] for r in records]) for name in STREAMS},
        }
        (output / "recognition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        mean = dict(summary)
        mean["physical_metrics"] = {}
        for key in (
            "fingertip_mean_error_mm",
            "excessive_contact_frame_ratio_percent",
            "fingertip_jerk_rms_m_s3",
            "contact_stall_frame_ratio_percent",
            "penetration.frame_ratio_percent",
            "penetration.mean_contact_depth_mm",
            "capsule_overlap_frame_ratio_percent",
        ):
            values = []
            for result in physical_reports:
                value = result
                for part in key.split("."):
                    value = value.get(part) if isinstance(value, dict) else None
                if value is not None and np.isfinite(value):
                    values.append(value)
            mean["physical_metrics"][key] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values)) if values else None,
                "valid_words": len(values),
                "total_words": len(records),
            }
        mean["aggregation"] = "Word-macro means and population standard deviations."
        (output / "mean_metrics.json").write_text(json.dumps(mean, indent=2, allow_nan=False) + "\n")
        top_path = output / "metrics.json"
        if top_path.exists():
            top = json.loads(top_path.read_text())
            top.setdefault("physical_metrics", {})["recognition"] = {
                "available": True,
                **summary["recognition"]["robot"],
                "by_stream": summary["recognition"],
                "provenance": provenance,
            }
            for item, report in zip(top.get("per_word", []), records):
                if item["word"] != report["expected_text"]:
                    raise ValueError("Dataset summary word ordering does not match recognition records.")
                item.setdefault("physical_metrics", {})["recognition"] = report
            top_path.write_text(json.dumps(top, indent=2, allow_nan=False) + "\n")
        with (output / "recognition_per_word.csv").open("w", newline="") as handle:
            fields = [
                "stream",
                "expected_text",
                "recognized_text",
                "character_accuracy_percent",
                "word_accuracy_percent",
                "edit_distance",
                "eos_reached",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for report in records:
                for name, result in report["streams"].items():
                    writer.writerow({"stream": name, **{key: result[key] for key in fields[1:]}})
        print(f"Saved recognition summary to {output}")
