"""Two-layer LSTM with short-term and long-term temporal attention heads."""

from __future__ import annotations

import torch
from torch import nn

from .features import FEATURE_DIM
from .labels import LONG_LABELS, SHORT_LABELS


class TemporalAttention(nn.Module):
    """Additive temporal attention over LSTM outputs."""

    def __init__(self, hidden_size: int, focus_last_n: int | None = None) -> None:
        super().__init__()
        self.focus_last_n = focus_last_n
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3:
            raise ValueError(f"Expected sequence shape (B, T, H), got {sequence.shape}")

        batch_size, total_steps, _ = sequence.shape
        if self.focus_last_n is not None and self.focus_last_n < total_steps:
            focused = sequence[:, -self.focus_last_n :, :]
            offset = total_steps - self.focus_last_n
        else:
            focused = sequence
            offset = 0

        scores = self.scorer(focused).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), focused).squeeze(1)

        full_weights = sequence.new_zeros(batch_size, total_steps)
        full_weights[:, offset:] = weights
        return context, full_weights


class IntentLSTMAttention(nn.Module):
    """Multi-task intent model with separate short and long attention branches."""

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        short_window: int = 20,
        use_attention: bool = True,
        num_short_classes: int = len(SHORT_LABELS),
        num_long_classes: int = len(LONG_LABELS),
    ) -> None:
        super().__init__()
        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "short_window": short_window,
            "use_attention": use_attention,
            "num_short_classes": num_short_classes,
            "num_long_classes": num_long_classes,
        }
        self.use_attention = use_attention

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.short_attention = TemporalAttention(hidden_size, focus_last_n=short_window)
        self.long_attention = TemporalAttention(hidden_size, focus_last_n=None)
        self.short_classifier = self._classifier(hidden_size, num_short_classes, dropout)
        self.long_classifier = self._classifier(hidden_size, num_long_classes, dropout)

    @staticmethod
    def _classifier(hidden_size: int, num_classes: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.input_norm(features)
        sequence, _ = self.lstm(x)

        if self.use_attention:
            short_context, short_attention = self.short_attention(sequence)
            long_context, long_attention = self.long_attention(sequence)
        else:
            short_steps = min(self.config["short_window"], sequence.shape[1])
            short_context = sequence[:, -short_steps:, :].mean(dim=1)
            long_context = sequence.mean(dim=1)
            short_attention = sequence.new_zeros(sequence.shape[0], sequence.shape[1])
            long_attention = sequence.new_zeros(sequence.shape[0], sequence.shape[1])

        return {
            "short_logits": self.short_classifier(short_context),
            "long_logits": self.long_classifier(long_context),
            "short_attention": short_attention,
            "long_attention": long_attention,
        }


def build_model_from_checkpoint_config(config: dict) -> IntentLSTMAttention:
    return IntentLSTMAttention(
        input_size=int(config.get("input_size", FEATURE_DIM)),
        hidden_size=int(config.get("hidden_size", 128)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.3)),
        short_window=int(config.get("short_window", 20)),
        use_attention=bool(config.get("use_attention", True)),
        num_short_classes=int(config.get("num_short_classes", len(SHORT_LABELS))),
        num_long_classes=int(config.get("num_long_classes", len(LONG_LABELS))),
    )
