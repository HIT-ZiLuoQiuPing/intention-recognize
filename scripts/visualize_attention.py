#!/usr/bin/env python3
"""Visualize short-term and long-term attention weights for one saved sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from intent_recognition.features import validate_feature_sequence
from intent_recognition.labels import long_label_name, short_label_name
from intent_recognition.train_utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def main() -> None:
    import matplotlib.pyplot as plt

    args = parse_args()
    device = resolve_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    model.eval()

    features = np.load(args.sample).astype(np.float32)
    validate_feature_sequence(features)
    batch = torch.from_numpy(features).unsqueeze(0).to(device=device, dtype=torch.float32)
    outputs = model(batch)

    short_probs = torch.softmax(outputs["short_logits"], dim=1)[0]
    long_probs = torch.softmax(outputs["long_logits"], dim=1)[0]
    short_id = int(short_probs.argmax().item())
    long_id = int(long_probs.argmax().item())
    short_attention = outputs["short_attention"][0].cpu().numpy()
    long_attention = outputs["long_attention"][0].cpu().numpy()
    frames = np.arange(len(short_attention))

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.plot(frames, long_attention, label="Long attention", linewidth=2)
    ax.plot(frames, short_attention, label="Short attention", linewidth=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Attention weight")
    ax.set_title(
        f"Predicted: {short_label_name(short_id)} / {long_label_name(long_id)} "
        f"({float(short_probs[short_id]):.2f}, {float(long_probs[long_id]):.2f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Saved attention visualization to {args.output}")


if __name__ == "__main__":
    main()
