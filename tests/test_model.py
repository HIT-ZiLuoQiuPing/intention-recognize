from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from intent_recognition.features import FEATURE_DIM, TARGET_FRAMES
from intent_recognition.model import IntentLSTMAttention


def test_model_forward_shapes():
    model = IntentLSTMAttention(input_size=FEATURE_DIM, hidden_size=16, num_layers=2, short_window=20)
    x = torch.randn(4, TARGET_FRAMES, FEATURE_DIM)
    outputs = model(x)

    assert outputs["short_logits"].shape == (4, 3)
    assert outputs["long_logits"].shape == (4, 3)
    assert outputs["short_attention"].shape == (4, TARGET_FRAMES)
    assert outputs["long_attention"].shape == (4, TARGET_FRAMES)
