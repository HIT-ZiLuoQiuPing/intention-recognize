"""Evaluation helpers for multi-task intent classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    return float((predictions == labels).mean())


def confusion_matrix(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels.astype(int), predictions.astype(int), strict=False):
        matrix[label, pred] += 1
    return matrix


def classification_metrics(
    short_predictions: np.ndarray,
    short_labels: np.ndarray,
    long_predictions: np.ndarray,
    long_labels: np.ndarray,
    num_short_classes: int,
    num_long_classes: int,
) -> dict:
    short_predictions = np.asarray(short_predictions)
    short_labels = np.asarray(short_labels)
    long_predictions = np.asarray(long_predictions)
    long_labels = np.asarray(long_labels)
    return {
        "short_accuracy": accuracy(short_predictions, short_labels),
        "long_accuracy": accuracy(long_predictions, long_labels),
        "combined_accuracy": float(
            ((short_predictions == short_labels) & (long_predictions == long_labels)).mean()
        )
        if short_labels.size
        else 0.0,
        "short_confusion_matrix": confusion_matrix(
            short_predictions, short_labels, num_short_classes
        ).tolist(),
        "long_confusion_matrix": confusion_matrix(
            long_predictions, long_labels, num_long_classes
        ).tolist(),
    }


def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], title: str, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = matrix.max() / 2 if matrix.size and matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
