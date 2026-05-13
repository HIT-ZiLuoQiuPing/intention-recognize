"""Training and checkpoint helpers."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .eval_utils import classification_metrics
from .labels import LONG_LABELS, SHORT_LABELS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = batch["features"].to(device=device, dtype=torch.float32)
    short_labels = batch["short_label"].to(device=device)
    long_labels = batch["long_label"].to(device=device)
    return features, short_labels, long_labels


def compute_loss(outputs: dict[str, torch.Tensor], short_labels: torch.Tensor, long_labels: torch.Tensor) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs["short_logits"], short_labels) + criterion(outputs["long_logits"], long_labels)


def train_one_epoch(model, loader, optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        features, short_labels, long_labels = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features)
        loss = compute_loss(outputs, short_labels, long_labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = features.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return {"loss": total_loss / max(total_items, 1)}


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_items = 0
    short_preds: list[np.ndarray] = []
    short_targets: list[np.ndarray] = []
    long_preds: list[np.ndarray] = []
    long_targets: list[np.ndarray] = []
    latencies: list[float] = []

    for batch in loader:
        features, short_labels, long_labels = batch_to_device(batch, device)
        start = time.perf_counter()
        outputs = model(features)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) / max(features.shape[0], 1))
        loss = compute_loss(outputs, short_labels, long_labels)

        batch_size = features.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        short_preds.append(outputs["short_logits"].argmax(dim=1).cpu().numpy())
        short_targets.append(short_labels.cpu().numpy())
        long_preds.append(outputs["long_logits"].argmax(dim=1).cpu().numpy())
        long_targets.append(long_labels.cpu().numpy())

    short_pred_arr = np.concatenate(short_preds) if short_preds else np.empty(0, dtype=np.int64)
    short_target_arr = np.concatenate(short_targets) if short_targets else np.empty(0, dtype=np.int64)
    long_pred_arr = np.concatenate(long_preds) if long_preds else np.empty(0, dtype=np.int64)
    long_target_arr = np.concatenate(long_targets) if long_targets else np.empty(0, dtype=np.int64)

    metrics = classification_metrics(
        short_pred_arr,
        short_target_arr,
        long_pred_arr,
        long_target_arr,
        num_short_classes=len(SHORT_LABELS),
        num_long_classes=len(LONG_LABELS),
    )
    metrics["loss"] = total_loss / max(total_items, 1)
    metrics["avg_latency_ms"] = float(np.mean(latencies) * 1000.0) if latencies else 0.0
    return metrics


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    epoch: int,
    config: dict,
    metrics: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "model_config": model.config,
            "train_config": config,
            "metrics": metrics,
            "short_labels": SHORT_LABELS,
            "long_labels": LONG_LABELS,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device):
    from .model import build_model_from_checkpoint_config

    checkpoint = torch.load(path, map_location=device)
    model = build_model_from_checkpoint_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, checkpoint
