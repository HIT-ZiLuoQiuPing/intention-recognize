#!/usr/bin/env python3
"""Run real-time webcam intent recognition with a sliding keypoint window."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
import time

import numpy as np
import torch

from intent_recognition.features import TARGET_FRAMES, preprocess_keypoint_sequence
from intent_recognition.labels import long_label_name, short_label_name
from intent_recognition.mediapipe_extractor import MediaPipeKeypointExtractor
from intent_recognition.train_utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--min-frames", type=int, default=60)
    parser.add_argument("--predict-every", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--stable-count", type=int, default=3)
    parser.add_argument("--max-missing-ratio", type=float, default=0.6)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def put_text(frame, text: str, y: int, color=(0, 255, 255)) -> None:
    import cv2

    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)


@torch.no_grad()
def predict(model, raw_window: list[np.ndarray], device: torch.device, max_missing_ratio: float) -> tuple[str, str, float]:
    features, _ = preprocess_keypoint_sequence(
        np.stack(raw_window, axis=0),
        target_frames=TARGET_FRAMES,
        max_missing_ratio=max_missing_ratio,
    )
    batch = torch.from_numpy(features).unsqueeze(0).to(device=device, dtype=torch.float32)
    outputs = model(batch)
    short_prob = torch.softmax(outputs["short_logits"], dim=1)[0]
    long_prob = torch.softmax(outputs["long_logits"], dim=1)[0]
    short_id = int(short_prob.argmax().item())
    long_id = int(long_prob.argmax().item())
    confidence = min(float(short_prob[short_id].item()), float(long_prob[long_id].item()))
    return short_label_name(short_id), long_label_name(long_id), confidence


def stable_prediction(recent: deque[tuple[str, str, float]], threshold: float, stable_count: int) -> tuple[str, str, float] | None:
    eligible = [item for item in recent if item[2] >= threshold]
    if len(eligible) < stable_count:
        return None
    last = eligible[-stable_count:]
    votes = Counter((short_name, long_name) for short_name, long_name, _ in last)
    (short_name, long_name), count = votes.most_common(1)[0]
    if count == stable_count:
        return short_name, long_name, min(conf for _, _, conf in last)
    return None


def main() -> None:
    import cv2

    args = parse_args()
    device = resolve_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    model.eval()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    raw_window: deque[np.ndarray] = deque(maxlen=TARGET_FRAMES)
    recent_predictions: deque[tuple[str, str, float]] = deque(maxlen=max(args.stable_count, 3))
    frame_idx = 0
    latest: tuple[str, str, float] | None = None
    confirmed: tuple[str, str, float] | None = None

    with MediaPipeKeypointExtractor() as extractor:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Could not read from camera")

                key_frame = extractor.process_bgr(frame)
                raw_window.append(key_frame.coords)
                frame_idx += 1

                if len(raw_window) >= args.min_frames and frame_idx % args.predict_every == 0:
                    start = time.perf_counter()
                    try:
                        latest = predict(model, list(raw_window), device, args.max_missing_ratio)
                        recent_predictions.append(latest)
                        stable = stable_prediction(recent_predictions, args.confidence, args.stable_count)
                        if stable is not None:
                            confirmed = stable
                    except ValueError:
                        latest = None
                    latency_ms = (time.perf_counter() - start) * 1000.0
                else:
                    latency_ms = 0.0

                display = extractor.draw(frame, key_frame.results)
                put_text(display, f"Window: {len(raw_window)}/{TARGET_FRAMES}", 30)
                if latest is not None:
                    put_text(display, f"Short: {latest[0]}  Long: {latest[1]}  Conf: {latest[2]:.2f}", 60)
                if confirmed is not None:
                    put_text(display, f"Confirmed: {confirmed[0]} + {confirmed[1]} ({confirmed[2]:.2f})", 90, (0, 255, 0))
                if latency_ms:
                    put_text(display, f"Latency: {latency_ms:.1f} ms", 120)
                put_text(display, "q: quit", 150)

                cv2.imshow("realtime_infer", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
