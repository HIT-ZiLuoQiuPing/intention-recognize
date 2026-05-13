# Human Intent Recognition with MediaPipe, LSTM, and Attention

This project implements a lightweight undergraduate-design friendly pipeline for recognizing human arm-operation intent:

- MediaPipe extracts right-arm and right-hand keypoints.
- Preprocessing normalizes keypoints and adds velocity features.
- A two-layer LSTM learns temporal motion patterns.
- Two attention heads classify short-term and long-term intent.

The first supported intents are:

| Action key | Short-term intent | Long-term intent |
| --- | --- | --- |
| `idle` | `idle` | `idle` |
| `reach_grasp` | `reach` | `grasp_intent` |
| `point_place` | `point` | `place_intent` |

## Installation

Use Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you only want to inspect the code or run static checks, installing the heavy camera dependencies is not required. `torch`, `opencv-python`, and `mediapipe` are needed for training and camera scripts.

## Dataset Collection

Collect one action class at a time. A sample is recorded for 3 seconds and saved as a processed `.npy` feature sequence with shape `(90, 144)`.

```bash
python3 scripts/collect_dataset.py --action reach_grasp --samples 100
python3 scripts/collect_dataset.py --action point_place --samples 100
python3 scripts/collect_dataset.py --action idle --samples 50
```

During collection:

- Press `Space` to record the next sample.
- Press `q` to quit.
- Keep the camera, background, and right hand visible.
- Vary speed, direction, and distance across samples.

Collected data is written to:

```text
data/sequences/
data/metadata.csv
```

## Dataset Summary

```bash
python3 scripts/summarize_dataset.py --metadata data/metadata.csv
```

## Training

```bash
python3 scripts/train_model.py \
  --metadata data/metadata.csv \
  --output-dir runs/intent_lstm \
  --epochs 100 \
  --batch-size 32
```

The script creates deterministic train/validation/test splits, trains with early stopping, and saves:

```text
runs/intent_lstm/best_model.pt
runs/intent_lstm/history.json
runs/intent_lstm/test_metrics.json
runs/intent_lstm/splits/
```

## Evaluation

```bash
python3 scripts/evaluate_model.py \
  --checkpoint runs/intent_lstm/best_model.pt \
  --metadata runs/intent_lstm/splits/test.csv \
  --output-dir runs/intent_lstm/eval
```

## Attention Visualization

```bash
python3 scripts/visualize_attention.py \
  --checkpoint runs/intent_lstm/best_model.pt \
  --sample data/sequences/reach_grasp/reach_grasp_0001.npy \
  --output runs/intent_lstm/attention_sample.png
```

## Real-Time Inference

```bash
python3 scripts/realtime_infer.py \
  --checkpoint runs/intent_lstm/best_model.pt
```

The real-time script maintains a sliding window and confirms an intent only when the same prediction is stable for several consecutive model calls.

## Project Structure

```text
intent_recognition/
  dataset.py              Dataset loading, metadata parsing, augmentation
  eval_utils.py           Accuracy, confusion matrix, metric helpers
  features.py             Keypoint interpolation, normalization, feature extraction
  labels.py               Intent label definitions
  mediapipe_extractor.py  MediaPipe wrapper for right-arm/right-hand points
  model.py                Two-layer LSTM with short/long attention heads
  train_utils.py          Training and checkpoint helpers
scripts/
  collect_dataset.py      Camera-based sample collection
  train_model.py          Model training
  evaluate_model.py       Offline evaluation
  realtime_infer.py       Online camera inference
  summarize_dataset.py    Metadata class counts
  visualize_attention.py  Attention-weight plot
tests/
  test_features.py
  test_model.py
```

## Notes for Thesis Experiments

Useful comparison experiments:

- LSTM with attention vs. LSTM without attention.
- One-layer LSTM vs. two-layer LSTM.
- Short-term accuracy, long-term accuracy, and combined accuracy.
- Confusion matrices for both output heads.
- Average real-time inference latency.
- Attention-weight visualization over the 90-frame sequence.

Example ablations:

```bash
python3 scripts/train_model.py --metadata data/metadata.csv --output-dir runs/no_attention --no-attention
python3 scripts/train_model.py --metadata data/metadata.csv --output-dir runs/one_layer --num-layers 1
```
