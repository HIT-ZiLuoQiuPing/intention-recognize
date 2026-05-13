"""Dataset loading, metadata parsing, and lightweight sequence augmentation."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .features import FEATURE_DIM, TARGET_FRAMES, validate_feature_sequence
from .labels import LONG_LABELS, SHORT_LABELS


@dataclass(frozen=True)
class MetadataRow:
    path: Path
    action: str
    short_label: int
    short_name: str
    long_label: int
    long_name: str


def _label_id(value: str, table: dict[str, int]) -> int:
    value = str(value)
    if value.isdigit():
        return int(value)
    return table[value]


def load_metadata(metadata_path: str | Path) -> list[MetadataRow]:
    """Load metadata CSV created by the collection script."""

    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    rows: list[MetadataRow] = []
    with metadata_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "action", "short_label", "short_name", "long_label", "long_name"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

        for raw_row in reader:
            sample_path = Path(raw_row["path"])
            if not sample_path.is_absolute():
                sample_path = (metadata_path.parent / sample_path).resolve()
            rows.append(
                MetadataRow(
                    path=sample_path,
                    action=raw_row["action"],
                    short_label=_label_id(raw_row["short_label"], SHORT_LABELS),
                    short_name=raw_row["short_name"],
                    long_label=_label_id(raw_row["long_label"], LONG_LABELS),
                    long_name=raw_row["long_name"],
                )
            )

    if not rows:
        raise ValueError(f"No rows found in metadata file: {metadata_path}")
    return rows


def write_metadata(rows: Iterable[MetadataRow], metadata_path: str | Path) -> None:
    """Write metadata rows to a CSV file."""

    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["path", "action", "short_label", "short_name", "long_label", "long_name"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": str(row.path),
                    "action": row.action,
                    "short_label": row.short_label,
                    "short_name": row.short_name,
                    "long_label": row.long_label,
                    "long_name": row.long_name,
                }
            )


def stratified_split(
    rows: list[MetadataRow],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[MetadataRow], list[MetadataRow], list[MetadataRow]]:
    """Split by action class so every split keeps roughly the same class balance."""

    if not 0.0 < train_ratio < 1.0 or not 0.0 <= val_ratio < 1.0:
        raise ValueError("train_ratio and val_ratio must be within [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    rng = random.Random(seed)
    by_action: dict[str, list[MetadataRow]] = {}
    for row in rows:
        by_action.setdefault(row.action, []).append(row)

    train: list[MetadataRow] = []
    val: list[MetadataRow] = []
    test: list[MetadataRow] = []

    for action_rows in by_action.values():
        shuffled = action_rows[:]
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = max(1, int(round(n_total * train_ratio))) if n_total >= 3 else max(0, n_total - 2)
        n_val = max(1, int(round(n_total * val_ratio))) if n_total >= 3 else 1 if n_total >= 2 else 0
        if n_train + n_val >= n_total and n_total > 1:
            n_train = max(1, n_total - 2)
            n_val = 1

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def augment_sequence(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply mild feature-space augmentation without changing sequence shape."""

    seq = np.asarray(sequence, dtype=np.float32).copy()

    if rng.random() < 0.7:
        seq += rng.normal(0.0, 0.01, size=seq.shape).astype(np.float32)

    if rng.random() < 0.5:
        shift = int(rng.integers(-5, 6))
        if shift > 0:
            seq = np.concatenate([np.repeat(seq[:1], shift, axis=0), seq[:-shift]], axis=0)
        elif shift < 0:
            seq = np.concatenate([seq[-shift:], np.repeat(seq[-1:], -shift, axis=0)], axis=0)

    if rng.random() < 0.3:
        drop_mask = rng.random(seq.shape[0]) < 0.03
        for idx in np.flatnonzero(drop_mask):
            seq[idx] = seq[idx - 1] if idx > 0 else seq[idx + 1]

    return seq.astype(np.float32)


class IntentSequenceDataset:
    """PyTorch-compatible dataset for saved intent feature sequences."""

    def __init__(
        self,
        rows: list[MetadataRow],
        augment: bool = False,
        target_frames: int = TARGET_FRAMES,
        seed: int = 42,
    ) -> None:
        self.rows = rows
        self.augment = augment
        self.target_frames = target_frames
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        features = np.load(row.path).astype(np.float32)
        validate_feature_sequence(features, target_frames=self.target_frames)
        if self.augment:
            features = augment_sequence(features, self.rng)
        if features.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM} features, got {features.shape[1]}")

        return {
            "features": torch.from_numpy(features),
            "short_label": torch.tensor(row.short_label, dtype=torch.long),
            "long_label": torch.tensor(row.long_label, dtype=torch.long),
            "path": str(row.path),
        }
