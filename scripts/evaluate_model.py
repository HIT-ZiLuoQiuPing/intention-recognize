#!/usr/bin/env python3
"""Evaluate a trained checkpoint on a metadata CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from intent_recognition.dataset import IntentSequenceDataset, load_metadata
from intent_recognition.eval_utils import plot_confusion_matrix, save_json
from intent_recognition.labels import LONG_LABELS, SHORT_LABELS
from intent_recognition.train_utils import evaluate, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eval"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    rows = load_metadata(args.metadata)
    loader = DataLoader(IntentSequenceDataset(rows), batch_size=args.batch_size, shuffle=False)
    metrics = evaluate(model, loader, device)
    save_json(metrics, args.output_dir / "metrics.json")

    plot_confusion_matrix(
        metrics["short_confusion_matrix"],
        list(SHORT_LABELS.keys()),
        "Short-term Intent Confusion Matrix",
        args.output_dir / "short_confusion_matrix.png",
    )
    plot_confusion_matrix(
        metrics["long_confusion_matrix"],
        list(LONG_LABELS.keys()),
        "Long-term Intent Confusion Matrix",
        args.output_dir / "long_confusion_matrix.png",
    )

    print(f"Short accuracy: {metrics['short_accuracy']:.4f}")
    print(f"Long accuracy: {metrics['long_accuracy']:.4f}")
    print(f"Combined accuracy: {metrics['combined_accuracy']:.4f}")
    print(f"Average latency: {metrics['avg_latency_ms']:.2f} ms/sample")


if __name__ == "__main__":
    main()
