#!/usr/bin/env python3
"""Train the two-layer LSTM + attention intent model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from intent_recognition.dataset import IntentSequenceDataset, load_metadata, stratified_split, write_metadata
from intent_recognition.eval_utils import plot_confusion_matrix, save_json
from intent_recognition.labels import LONG_LABELS, SHORT_LABELS
from intent_recognition.model import IntentLSTMAttention
from intent_recognition.train_utils import evaluate, save_checkpoint, set_seed, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/intent_lstm"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--short-window", type=int, default=20)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = args.output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metadata(args.metadata)
    train_rows, val_rows, test_rows = stratified_split(rows, seed=args.seed)
    write_metadata(train_rows, split_dir / "train.csv")
    write_metadata(val_rows, split_dir / "val.csv")
    write_metadata(test_rows, split_dir / "test.csv")

    train_dataset = IntentSequenceDataset(train_rows, augment=True, seed=args.seed)
    val_dataset = IntentSequenceDataset(val_rows, augment=False, seed=args.seed)
    test_dataset = IntentSequenceDataset(test_rows, augment=False, seed=args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = resolve_device(args.device)
    model = IntentLSTMAttention(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        short_window=args.short_window,
        use_attention=not args.no_attention,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_config = vars(args) | {
        "device": str(device),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_test": len(test_rows),
    }
    save_json(train_config, args.output_dir / "train_config.json")

    best_val = -1.0
    stale_epochs = 0
    history: list[dict] = []
    best_path = args.output_dir / "best_model.pt"

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        score = (val_metrics["short_accuracy"] + val_metrics["long_accuracy"]) / 2.0
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_short_accuracy": val_metrics["short_accuracy"],
            "val_long_accuracy": val_metrics["long_accuracy"],
            "val_combined_accuracy": val_metrics["combined_accuracy"],
        }
        history.append(row)
        save_json({"history": history}, args.output_dir / "history.json")

        if score > best_val:
            best_val = score
            stale_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, train_config, val_metrics)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best validation score={best_val:.4f}")
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    save_json(test_metrics, args.output_dir / "test_metrics.json")

    short_names = list(SHORT_LABELS.keys())
    long_names = list(LONG_LABELS.keys())
    plot_confusion_matrix(
        test_metrics["short_confusion_matrix"],
        short_names,
        "Short-term Intent Confusion Matrix",
        args.output_dir / "short_confusion_matrix.png",
    )
    plot_confusion_matrix(
        test_metrics["long_confusion_matrix"],
        long_names,
        "Long-term Intent Confusion Matrix",
        args.output_dir / "long_confusion_matrix.png",
    )

    print(f"Best checkpoint: {best_path}")
    print(f"Test short accuracy: {test_metrics['short_accuracy']:.4f}")
    print(f"Test long accuracy: {test_metrics['long_accuracy']:.4f}")
    print(f"Average latency: {test_metrics['avg_latency_ms']:.2f} ms/sample")


if __name__ == "__main__":
    main()
