#!/usr/bin/env python3
"""Collect intent samples from a webcam and save preprocessed feature sequences."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from intent_recognition.features import TARGET_FRAMES, SequenceQualityError, preprocess_keypoint_sequence
from intent_recognition.labels import ACTION_TO_LABELS, labels_for_action
from intent_recognition.mediapipe_extractor import MediaPipeKeypointExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=sorted(ACTION_TO_LABELS))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sequences"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--max-missing-ratio", type=float, default=0.2)
    parser.add_argument("--countdown", type=float, default=1.0)
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def append_metadata(metadata_path: Path, row: dict) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    exists = metadata_path.exists()
    with metadata_path.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "path",
            "action",
            "short_label",
            "short_name",
            "long_label",
            "long_name",
            "missing_ratio",
            "original_frames",
            "created_at",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def put_text(frame, text: str, y: int) -> None:
    import cv2

    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main() -> None:
    import cv2

    args = parse_args()
    labels = labels_for_action(args.action)
    target_raw_frames = max(1, int(round(args.seconds * args.fps)))
    action_dir = args.output_dir / args.action
    action_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    saved = 0
    attempted = 0
    frame_interval = 1.0 / max(args.fps, 1)

    print(f"Collecting action={args.action}; target saved samples={args.samples}")
    print("Press Space to record a sample; press q to quit.")

    with MediaPipeKeypointExtractor() as extractor:
        try:
            while saved < args.samples:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Could not read from camera")

                key_frame = extractor.process_bgr(frame)
                display = extractor.draw(frame, key_frame.results)
                put_text(display, f"Action: {args.action}  Saved: {saved}/{args.samples}", 30)
                put_text(display, "Space: record   q: quit", 60)

                if not args.no_preview:
                    cv2.imshow("collect_dataset", display)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = ord(" ")

                if key == ord("q"):
                    break
                if key != ord(" "):
                    continue

                if args.countdown > 0:
                    time.sleep(args.countdown)

                raw_frames: list[np.ndarray] = []
                start = time.perf_counter()
                next_frame_at = start
                while len(raw_frames) < target_raw_frames:
                    now = time.perf_counter()
                    if now < next_frame_at:
                        time.sleep(min(0.005, next_frame_at - now))
                        continue

                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Could not read from camera during recording")
                    key_frame = extractor.process_bgr(frame)
                    raw_frames.append(key_frame.coords)
                    next_frame_at += frame_interval

                    if not args.no_preview:
                        display = extractor.draw(frame, key_frame.results)
                        put_text(display, f"Recording {len(raw_frames)}/{target_raw_frames}", 30)
                        cv2.imshow("collect_dataset", display)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                attempted += 1
                raw_sequence = np.stack(raw_frames, axis=0)
                try:
                    features, report = preprocess_keypoint_sequence(
                        raw_sequence,
                        target_frames=TARGET_FRAMES,
                        max_missing_ratio=args.max_missing_ratio,
                    )
                except SequenceQualityError as exc:
                    print(f"Discarded sample {attempted}: {exc}")
                    continue

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{args.action}_{timestamp}_{saved + 1:04d}.npy"
                sample_path = action_dir / filename
                np.save(sample_path, features)
                append_metadata(
                    args.metadata,
                    {
                        "path": sample_path,
                        "action": args.action,
                        "short_label": labels.short_id,
                        "short_name": labels.short_name,
                        "long_label": labels.long_id,
                        "long_name": labels.long_name,
                        "missing_ratio": f"{report.missing_ratio:.6f}",
                        "original_frames": report.original_frames,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                saved += 1
                print(f"Saved {sample_path}  missing_ratio={report.missing_ratio:.3f}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    print(f"Done. Saved {saved} samples for action={args.action}.")


if __name__ == "__main__":
    main()
