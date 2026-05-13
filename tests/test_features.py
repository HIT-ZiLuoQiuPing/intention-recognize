from __future__ import annotations

import numpy as np

from intent_recognition.features import FEATURE_DIM, TARGET_FRAMES, preprocess_keypoint_sequence


def test_preprocess_keypoint_sequence_shape_and_finiteness():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(72, 24, 3)).astype(np.float32)
    raw[10:12, 5, :] = np.nan

    features, report = preprocess_keypoint_sequence(raw, max_missing_ratio=0.2)

    assert features.shape == (TARGET_FRAMES, FEATURE_DIM)
    assert np.isfinite(features).all()
    assert report.original_frames == 72


def test_preprocess_rejects_too_many_missing_values():
    raw = np.full((90, 24, 3), np.nan, dtype=np.float32)

    try:
        preprocess_keypoint_sequence(raw, max_missing_ratio=0.2)
    except ValueError as exc:
        assert "Missing keypoint ratio" in str(exc)
    else:
        raise AssertionError("Expected missing sequence to be rejected")
