"""Feature extraction from right-arm and right-hand MediaPipe keypoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TARGET_FRAMES = 90
KEYPOINT_COUNT = 24
COORD_DIM = 3
FEATURE_DIM = KEYPOINT_COUNT * COORD_DIM * 2

RIGHT_SHOULDER_INDEX = 0
RIGHT_ELBOW_INDEX = 1
RIGHT_WRIST_INDEX = 2


@dataclass(frozen=True)
class PreprocessReport:
    missing_ratio: float
    original_frames: int
    target_frames: int
    scale: float


class SequenceQualityError(ValueError):
    """Raised when a recorded sequence is too incomplete to keep."""


def missing_ratio(raw_coords: np.ndarray) -> float:
    """Return the proportion of non-finite coordinate values."""

    coords = np.asarray(raw_coords, dtype=np.float32)
    if coords.size == 0:
        return 1.0
    return float((~np.isfinite(coords)).sum() / coords.size)


def interpolate_nans(raw_coords: np.ndarray) -> np.ndarray:
    """Fill missing coordinates along the temporal axis with interpolation."""

    coords = np.asarray(raw_coords, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[1:] != (KEYPOINT_COUNT, COORD_DIM):
        raise ValueError(
            f"Expected raw keypoints with shape (T, {KEYPOINT_COUNT}, {COORD_DIM}), "
            f"got {coords.shape}"
        )

    filled = coords.copy()
    frames = np.arange(filled.shape[0], dtype=np.float32)

    for point_idx in range(filled.shape[1]):
        for coord_idx in range(filled.shape[2]):
            values = filled[:, point_idx, coord_idx]
            valid = np.isfinite(values)
            if valid.all():
                continue
            if not valid.any():
                filled[:, point_idx, coord_idx] = 0.0
                continue
            if valid.sum() == 1:
                filled[:, point_idx, coord_idx] = values[valid][0]
                continue
            filled[:, point_idx, coord_idx] = np.interp(
                frames,
                frames[valid],
                values[valid],
            )

    return filled.astype(np.float32)


def resample_sequence(sequence: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """Resample a sequence along time to a fixed number of frames."""

    seq = np.asarray(sequence, dtype=np.float32)
    if seq.shape[0] == target_frames:
        return seq.copy()
    if seq.shape[0] < 1:
        raise ValueError("Cannot resample an empty sequence")

    old_t = np.linspace(0.0, 1.0, seq.shape[0], dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    flat = seq.reshape(seq.shape[0], -1)
    resampled = np.empty((target_frames, flat.shape[1]), dtype=np.float32)

    for feature_idx in range(flat.shape[1]):
        resampled[:, feature_idx] = np.interp(new_t, old_t, flat[:, feature_idx])

    return resampled.reshape((target_frames, *seq.shape[1:])).astype(np.float32)


def normalize_keypoints(coords: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float]:
    """Normalize coordinates by right shoulder origin and upper-arm length."""

    values = np.asarray(coords, dtype=np.float32)
    origin = values[:, RIGHT_SHOULDER_INDEX : RIGHT_SHOULDER_INDEX + 1, :]
    centered = values - origin

    upper_arm = np.linalg.norm(
        values[:, RIGHT_ELBOW_INDEX, :] - values[:, RIGHT_SHOULDER_INDEX, :],
        axis=-1,
    )
    valid_scale = upper_arm[np.isfinite(upper_arm) & (upper_arm > eps)]
    scale = float(np.median(valid_scale)) if valid_scale.size else 1.0
    if scale < eps:
        scale = 1.0

    return (centered / scale).astype(np.float32), scale


def add_velocity_features(normalized_coords: np.ndarray) -> np.ndarray:
    """Append first-order temporal velocity to normalized coordinates."""

    coords = np.asarray(normalized_coords, dtype=np.float32)
    velocity = np.diff(coords, axis=0, prepend=coords[:1])
    features = np.concatenate([coords, velocity], axis=-1)
    return features.reshape(coords.shape[0], -1).astype(np.float32)


def preprocess_keypoint_sequence(
    raw_coords: np.ndarray,
    target_frames: int = TARGET_FRAMES,
    max_missing_ratio: float | None = 0.2,
) -> tuple[np.ndarray, PreprocessReport]:
    """Convert raw MediaPipe keypoints into a fixed `(T, 144)` feature sequence."""

    raw = np.asarray(raw_coords, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (KEYPOINT_COUNT, COORD_DIM):
        raise ValueError(
            f"Expected raw keypoints with shape (T, {KEYPOINT_COUNT}, {COORD_DIM}), "
            f"got {raw.shape}"
        )

    ratio = missing_ratio(raw)
    if max_missing_ratio is not None and ratio > max_missing_ratio:
        raise SequenceQualityError(
            f"Missing keypoint ratio {ratio:.3f} exceeds max_missing_ratio={max_missing_ratio:.3f}"
        )

    filled = interpolate_nans(raw)
    fixed = resample_sequence(filled, target_frames=target_frames)
    normalized, scale = normalize_keypoints(fixed)
    features = add_velocity_features(normalized)
    report = PreprocessReport(
        missing_ratio=ratio,
        original_frames=int(raw.shape[0]),
        target_frames=int(target_frames),
        scale=scale,
    )
    return features, report


def validate_feature_sequence(features: np.ndarray, target_frames: int = TARGET_FRAMES) -> None:
    """Raise if a saved feature sequence has the wrong shape or invalid values."""

    values = np.asarray(features)
    expected_shape = (target_frames, FEATURE_DIM)
    if values.shape != expected_shape:
        raise ValueError(f"Expected feature shape {expected_shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Feature sequence contains NaN or infinite values")
