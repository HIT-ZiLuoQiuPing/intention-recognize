"""Shared label definitions for the intent-recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass


SHORT_LABELS: dict[str, int] = {
    "idle": 0,
    "reach": 1,
    "point": 2,
}

LONG_LABELS: dict[str, int] = {
    "idle": 0,
    "grasp_intent": 1,
    "place_intent": 2,
}

ACTION_TO_LABELS: dict[str, tuple[str, str]] = {
    "idle": ("idle", "idle"),
    "reach_grasp": ("reach", "grasp_intent"),
    "point_place": ("point", "place_intent"),
}

SHORT_ID_TO_LABEL = {value: key for key, value in SHORT_LABELS.items()}
LONG_ID_TO_LABEL = {value: key for key, value in LONG_LABELS.items()}


@dataclass(frozen=True)
class IntentLabels:
    action: str
    short_name: str
    short_id: int
    long_name: str
    long_id: int


def labels_for_action(action: str) -> IntentLabels:
    """Return label names and ids for a collection action key."""

    if action not in ACTION_TO_LABELS:
        allowed = ", ".join(sorted(ACTION_TO_LABELS))
        raise ValueError(f"Unknown action '{action}'. Expected one of: {allowed}")

    short_name, long_name = ACTION_TO_LABELS[action]
    return IntentLabels(
        action=action,
        short_name=short_name,
        short_id=SHORT_LABELS[short_name],
        long_name=long_name,
        long_id=LONG_LABELS[long_name],
    )


def short_label_name(label_id: int) -> str:
    return SHORT_ID_TO_LABEL[int(label_id)]


def long_label_name(label_id: int) -> str:
    return LONG_ID_TO_LABEL[int(label_id)]
