# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Physical-quality metrics of simulated robot trajectories (NumPy/SciPy only, no simulator).

Frames of the preparation hold are excluded. Reported per word: fingertip error [mm], contact and excessive-contact
frame ratios, fingertip jerk RMS [m/s^3], capsule-overlap frame ratio, contact-stall ratio, joint jerk and the
Newton self-penetration frame ratio and depths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import savgol_filter

# Fingertip landmarks in MediaPipe order: thumb, index, middle, ring, little.
TIPS = [4, 8, 12, 16, 20]

# Capsule proxy of finger self-overlap: landmark segments get Shadow Hand collision-capsule radii [m].
FINGERS = {"th": [2, 3, 4], "ff": [5, 6, 7, 8], "mf": [9, 10, 11, 12], "rf": [13, 14, 15, 16], "lf": [17, 18, 19, 20]}
# Proximal, middle, distal radii (thumb: middle, distal).
RADII = {"th": [0.011, 0.0092], "ff": [0.010, 0.008, 0.007], "mf": [0.010, 0.008, 0.007]}
RADII.update(rf=RADII["ff"], lf=RADII["ff"])
FINGER_PAIRS = [("ff", "mf"), ("mf", "rf"), ("rf", "lf"), ("th", "ff"), ("th", "mf")]


@dataclass
class MetricSettings:
    """Thresholds: forces [N], speeds [m/s], distances [m]; Savitzky-Golay window [frames]."""

    contact_force_n: float = 1.0
    excessive_force_n: float = 5.0
    target_speed_m_s: float = 0.02
    stall_speed_m_s: float = 0.005
    stall_error_m: float = 0.025
    penetration_tolerance_m: float = 0.001
    smoothing_window: int = 5


def _segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Exact closest distance [m] between segments p1-q1 and p2-q2, broadcast over leading axes [..., 3]."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    denom = a * e - b * b
    s = np.clip(np.divide(b * f - c * e, denom, out=np.zeros_like(denom), where=denom > 1e-12), 0, 1)
    t = (b * s + f) / e
    # Re-clamp t and recompute s on the boundary (Ericson, Real-Time Collision Detection 5.1.9).
    s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s))
    t = np.clip(t, 0, 1)
    return np.linalg.norm(p1 + d1 * s[..., None] - p2 - d2 * t[..., None], axis=-1)


def overlap_depth(points: np.ndarray) -> np.ndarray:
    """Per-frame maximum capsule overlap [m] of adjacent fingers for palm-local landmarks [T, 21, 3], shape [T]."""
    depths = []
    for a, b in FINGER_PAIRS:
        ia, ib = FINGERS[a], FINGERS[b]
        distance = _segment_distance(
            points[:, ia[:-1], None], points[:, ia[1:], None], points[:, None, ib[:-1]], points[:, None, ib[1:]]
        )
        radii = np.array(RADII[a])[:, None] + np.array(RADII[b])[None]
        depths.append((radii[None] - distance).max(axis=(1, 2)))
    return np.max(depths, axis=0)


def text_metrics(expected: str, predicted: str) -> dict:
    """Word accuracy and unit-cost edit-distance character accuracy [%] of one recognized word."""
    if not expected:
        raise ValueError("Expected text must be nonempty.")
    previous = list(range(len(predicted) + 1))
    for i, left in enumerate(expected, 1):
        current = [i]
        for j, right in enumerate(predicted, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    edits = previous[-1]
    return {
        "expected_text": expected,
        "recognized_text": predicted,
        "word_accuracy_percent": 100.0 * (expected == predicted),
        "character_accuracy_percent": 100.0 * (1 - edits / len(expected)),
        "edit_distance": edits,
        "note": "Edit accuracy may be negative when insertions exceed the reference length.",
    }


def source_motion_trajectory(data: dict) -> dict:
    """Drop the frames of the preparation hold and shift timestamps [s] so the source motion starts at 0."""
    preparation = float(np.asarray(data.get("preparation_duration_s", 0.0)).item())
    if not np.isfinite(preparation) or preparation < 0:
        raise ValueError("Preparation duration must be finite and nonnegative.")
    if preparation == 0:
        return data
    times = np.asarray(data["timestamps"])
    indices = np.flatnonzero(times > preparation + 1e-6)
    if len(indices) < 2:
        raise ValueError("At least two source-motion frames must remain after preparation.")
    start = int(indices[0])
    result = dict(data)
    for key in (
        "actual", "target", "timestamps", "errors", "actions", "max_contact_force_n",
        "joint_positions_rad", "joint_velocities_rad_s", "penetration_max_depth_m",
        "penetration_depth_sum_m", "penetration_contact_count",
    ):
        if key in data:
            result[key] = np.asarray(data[key])[start:]
    if "reported_contact_offsets" in data:
        offsets = np.asarray(data["reported_contact_offsets"])
        result["reported_contact_depth_m"] = np.asarray(data["reported_contact_depth_m"])[offsets[start]:]
        result["reported_contact_offsets"] = offsets[start:] - offsets[start]
    result["timestamps"] = result["timestamps"] - preparation
    result["preparation_duration_s"] = np.array(0.0)
    return result


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1 or len(values) < window:
        return values.copy()
    return savgol_filter(values, window, 3, axis=0, mode="interp")


def _rms_vectors(values: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(np.sum(values**2, axis=-1)))) if values.size else None


def compute_physical_metrics(data: dict, settings: MetricSettings | None = None) -> dict:
    """Measure tracking, contact, jerk [m/s^3], stalls and penetration of one trajectory.

    Args:
        data: Trajectory arrays: ``actual``/``target`` [m] [T, 21, 3], ``timestamps`` [s], ``max_contact_force_n``
            [N], optionally joint positions [rad] and reported contact depths [m]. Frames within
            ``preparation_duration_s`` [s] are excluded.
        settings: Thresholds and smoothing settings.

    Returns:
        JSON-compatible metrics with counts and denominators; unavailable values are null.
    """
    data = source_motion_trajectory(data)
    settings = settings or MetricSettings()
    actual, target, times, forces = (
        np.asarray(data[k]) for k in ("actual", "target", "timestamps", "max_contact_force_n")
    )
    n = len(times)
    if actual.shape != (n, 21, 3) or target.shape != actual.shape or forces.shape != (n,) or n < 2:
        raise ValueError("Expected at least two frames with matching trajectory and force arrays.")
    if not all(np.isfinite(v).all() for v in (actual, target, times, forces)) or np.any(forces < 0):
        raise ValueError("Trajectory must be finite and contact magnitudes nonnegative.")
    intervals = np.diff(times)
    if np.any(intervals <= 0) or not np.allclose(intervals, intervals[0], rtol=1e-4, atol=1e-7):
        raise ValueError("Metrics require strictly increasing, uniformly sampled timestamps.")
    dt = float(intervals[0])
    contact = forces > settings.contact_force_n
    excessive = forces > settings.excessive_force_n
    error = np.linalg.norm(actual[:, TIPS] - target[:, TIPS], axis=-1)
    a = _smooth(actual[:, TIPS], settings.smoothing_window)
    t = _smooth(target[:, TIPS], settings.smoothing_window)
    actual_speed = np.linalg.norm(np.diff(a, axis=0), axis=-1) / dt
    target_speed = np.linalg.norm(np.diff(t, axis=0), axis=-1) / dt
    stalled = (
        (target_speed > settings.target_speed_m_s)
        & (actual_speed < settings.stall_speed_m_s)
        & (error[1:] > settings.stall_error_m)
        & contact[1:, None]
    ).any(axis=1)
    jerk = np.diff(a, n=3, axis=0) / dt**3 if n >= 4 else np.empty((0, 5, 3))
    jerk_contact = np.array([contact[i : i + 4].any() for i in range(max(n - 3, 0))], dtype=bool)
    result = {
        "settings": asdict(settings),
        "sampling_dt_s": dt,
        "contact_frames": int(contact.sum()),
        "excessive_contact_frames": int(excessive.sum()),
        "excessive_contact_frame_ratio_percent": float(excessive.mean() * 100),
        "contact_frame_ratio_percent": float(contact.mean() * 100),
        "fingertip_mean_error_mm": float(error.mean() * 1000),
        "fingertip_max_error_mm": float(error.max() * 1000),
        "contact_fingertip_mean_error_mm": float(error[contact].mean() * 1000) if contact.any() else None,
        "fingertip_jerk_rms_m_s3": _rms_vectors(jerk),
        "capsule_overlap_frame_ratio_percent": float(
            (overlap_depth(actual) > settings.penetration_tolerance_m).mean() * 100
        ),
        "contact_fingertip_jerk_rms_m_s3": _rms_vectors(jerk[jerk_contact]),
        "jerk_samples": len(jerk),
        "contact_jerk_samples": int(jerk_contact.sum()),
        "contact_stall_frames": int(stalled.sum()),
        "contact_stall_frame_ratio_percent": float(stalled.mean() * 100),
        "stall_evaluable_frames": n - 1,
        "contact_stall_evaluable_frames": int(contact[1:].sum()),
        "contact_stall_ratio_within_contact_percent": float(stalled.sum() / contact[1:].sum() * 100)
        if contact[1:].any()
        else None,
        "contact_scope": "Any monitored finger-link normal-force history peak; not pairwise force or causal evidence.",
        "stall_scope": "Any fingertip stationary with moving target and large error during aggregate contact.",
        "derivative_method": (
            "Savitzky-Golay polynomial 3 (if enough frames), backward differences; jerk vector-norm RMS."
        ),
        "penetration": {"available": False, "reason": "No reported-contact depth samples in trajectory."},
        "recognition": {"available": False, "reason": "No robot-motion recognizer predictions provided."},
    }
    if "joint_positions_rad" in data:
        joints = np.asarray(data["joint_positions_rad"])
        if joints.ndim != 2 or joints.shape[0] != n or not np.isfinite(joints).all():
            raise ValueError("Invalid joint position samples.")
        joint_jerk = np.diff(_smooth(joints, settings.smoothing_window), n=3, axis=0) / dt**3
        result["joint_jerk_rms_rad_s3"] = float(np.sqrt(np.mean(joint_jerk**2))) if joint_jerk.size else None
    if "reported_contact_depth_m" in data:
        raw_depths = np.asarray(data["reported_contact_depth_m"])
        offsets = np.asarray(data["reported_contact_offsets"])
        if (
            offsets.shape != (n + 1,)
            or not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0
            or offsets[-1] != len(raw_depths)
            or np.any(np.diff(offsets) < 0)
            or not np.isfinite(raw_depths).all()
            or np.any(raw_depths < 0)
        ):
            raise ValueError("Invalid reported-contact depth offsets.")
        frames = [raw_depths[offsets[i] : offsets[i + 1]] for i in range(n)]
        depths = np.array([f.max() if len(f) else 0.0 for f in frames])
        penetrating = [f[f > settings.penetration_tolerance_m] for f in frames]
        sums = np.array([f.sum() for f in penetrating])
        counts = np.array([len(f) for f in penetrating])
        overlapping = depths > settings.penetration_tolerance_m
        result["penetration"] = {
            "available": True,
            "frame_ratio_percent": float(overlapping.mean() * 100),
            "frames": int(overlapping.sum()),
            "sampled_frames": n,
            "mean_contact_depth_mm": float(sums.sum() / counts.sum() * 1000) if counts.sum() else 0.0,
            "max_depth_mm": float(depths.max() * 1000),
            "penetrating_contact_samples": int(counts.sum()),
            "scope": (
                "Reported hand self-contact surface separation at control-step endpoints; not exhaustive mesh testing."
            ),
        }
    return result
