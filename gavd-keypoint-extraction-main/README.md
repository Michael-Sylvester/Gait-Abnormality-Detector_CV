# Gait Abnormality Detection via Spatio-Temporal Graph Convolutional Networks

This repository contains the full pipeline for our work on automated gait abnormality
detection using skeletal pose sequences. We extract COCO-17 keypoints from the GAVD
dataset using YOLOv8-Pose and train a Spatio-Temporal Graph Convolutional Network
(ST-GCN) to classify walking sequences as normal or abnormal. We also provide a
1D-CNN temporal baseline and a channel ablation study.

---

## Contents

- [Dataset](#dataset)
- [Setup](#setup)
- [Step 1 — Keypoint Extraction](#step-1--keypoint-extraction)
- [Step 2 — Training](#step-2--training)
  - [Baseline (1D-CNN)](#baseline-1d-cnn)
  - [ST-GCN](#st-gcn)
  - [Channel Ablation](#channel-ablation)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)

---

## Dataset

We use the **Gait Abnormality Video Dataset (GAVD)**, which contains 1,874 annotated
walking sequences covering 12 gait categories (normal, parkinsons, stroke, myopathic,
cerebral palsy, and others).

> **Download:** Link to GAVD dataset videos can be found in the csv files [here](https://github.com/Rahmyyy/GAVD/tree/main/data) 

After downloading, place the raw videos under `data/videos/` (subdirectories are fine).
The extracted keypoint tensors used for training are stored separately in
`gavd_handoff/data/keypoints/` — see [Step 1](#step-1--keypoint-extraction) to
regenerate them, or use our pre-extracted version linked above.

---

## Setup

Requires Python 3.10+.

```bash
git clone <repo-url>
cd gavd-keypoint-extraction

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install scikit-learn pandas matplotlib
```

The first extraction run will automatically download `yolov8m-pose.pt` (~50 MB)
to your Ultralytics cache.

---

## Step 1 — Keypoint Extraction

We use **YOLOv8-Pose** to extract COCO-17 skeletal keypoints from each video.

**Why YOLOv8-Pose:**
- Native COCO-17 output — directly compatible with ST-GCN implementations
- Built-in multi-person tracking — handles clips with passersby
- Single-stage detection and pose estimation in one forward pass
- Strong performance under partial occlusion

**Subject selection:** In multi-person frames, we select the track ID that appears
in the most frames, breaking ties by average bounding-box area (the gait subject
is typically closest to the camera). See `src/track.py`.

**Output format per sequence:**

| File | Description |
|---|---|
| `<seq_id>.npy` | `float32` tensor `(3, T, 17, 1)` — channels: x, y, confidence. Coordinates are hip-centred and torso-scaled. Missing frames are `NaN`. |
| `<seq_id>.json` | Human-readable metadata, raw pixel-space keypoints, joint names, skeleton edges |
| `_labels.csv` | Master index with binary label, fine-grained pathology label, and per-sequence detection rate |

### Run extraction

```bash
python -m src.extract_gavd \
    --meta data/gavd-meta/data \
    --videos data/videos \
    --output gavd_handoff/data/keypoints \
    --skip-existing
```

Switch model size (`n` = fastest, `x` = most accurate):

```bash
python -m src.extract_gavd ... --model yolov8x-pose.pt
```

Visually verify a sample before processing the full dataset:

```bash
python -m src.verify \
    --video data/videos/clip.mp4 \
    --json  gavd_handoff/data/keypoints/<seq_id>.json \
    --output samples/clip_overlay.mp4
```

---

## Step 2 — Training

### Data splits and preprocessing

All models use the same reproducible 80 / 10 / 10 train / val / test split,
partitioned by `video_id` using `GroupShuffleSplit`. Splitting by subject
identity (rather than randomly) is essential — random splitting allows the same
person to appear in both train and test, which inflates all metrics.

Sequences with `detection_rate < 0.50` are discarded, leaving **1,743 usable
sequences** from the original 1,779.

| Split | Sequences | Normal | Abnormal |
|---|---|---|---|
| Train | 1,431 | 207 | 1,224 |
| Val | 110 | — | — |
| Test | 202 | 13 | 189 |

**Class imbalance (6:1 abnormal:normal)** is handled by two complementary mechanisms:
- `WeightedRandomSampler` — each training batch is drawn with inverse-frequency
  per-sample weights, so the model sees a balanced class distribution every epoch
- `CrossEntropyLoss(weight=...)` — the loss function assigns higher penalty to
  errors on the minority Normal class

**Missing frames** are handled by replacing `NaN` with zero in the input tensor
and using confidence-weighted temporal pooling at the output: each frame is
weighted by its mean joint confidence score, so frames with no valid detection
contribute approximately zero to the sequence representation.

---

### Baseline (1D-CNN)

A temporal CNN that treats gait as a time series of flattened joint coordinates
with no awareness of skeleton topology. Its role is to establish a floor:
any improvement from ST-GCN over this baseline can be attributed to the graph
structure capturing joint-to-joint spatial relationships.

**Architecture:** `(B, 34, 150)` → Conv1d(34→64→128→256, k=3) → confidence-weighted
temporal pool → Linear(256→2)

```bash
python -m baseline.train \
    --keypoint_dir gavd_handoff/data/keypoints \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3
```

Checkpoint and logs are saved to `baseline/runs/`.

---

### ST-GCN

Spatio-Temporal Graph Convolutional Network (Yan et al., 2018), adapted for the
COCO-17 skeleton and binary gait classification.

**Graph construction:** The COCO-17 skeleton provides 16 anatomical edges. Two
additional edges (nose → left/right shoulder) are added to connect the otherwise
isolated head joints to the body. The adjacency matrix uses the spatial configuration
partitioning (3 subsets: self, centripetal, centrifugal), with a learnable per-edge
importance weight in each block. The graph root is the left hip — the most stable
anatomical landmark during gait.

**Architecture:** 9 ST-GCN blocks over `(B, 3, 300, 17)` input tensors, with
temporal stride-2 downsampling at blocks 4 and 7, followed by confidence-weighted
temporal pooling and a linear classifier. ~3.0M parameters.

#### Train locally

```bash
python -m stgcn.train \
    --keypoint_dir gavd_handoff/data/keypoints \
    --epochs 80 \
    --batch_size 16 \
    --lr 1e-3
```

#### Train on GPU via Modal

```bash
# one-time: upload keypoint data to Modal volume
modal run modal_train.py --upload

# train
modal run modal_train.py

# resume an interrupted run
modal run modal_train.py --resume

# download results
modal volume get gavd-runs best_model.pt      stgcn/runs/best_model.pt
modal volume get gavd-runs history.json       stgcn/runs/history.json
modal volume get gavd-runs test_results.json  stgcn/runs/test_results.json
```

Checkpoints are saved to `stgcn/runs/`. The best checkpoint (by val macro F1)
is stored as `best_model.pt`; `last_checkpoint.pt` supports resumption.

---

### Channel Ablation

Trains ST-GCN with only (x, y) coordinates as input — dropping the confidence
channel — to quantify how much the per-joint confidence score contributes beyond
its role in weighted temporal pooling.

```bash
# Modal
modal run modal_train.py --ablate

# local
python -m stgcn.train \
    --keypoint_dir gavd_handoff/data/keypoints \
    --out_dir stgcn/runs/ablation \
    --epochs 80 --batch_size 16 --lr 1e-3 \
    --in_channels 2
```

Results are saved to `stgcn/runs/ablation/` and do not overwrite the main run.

---

## Evaluation

After training, find the optimal decision threshold on the val set and evaluate
on the test set. The threshold is tuned on val to maximise macro F1; the test set
is evaluated only once at the chosen threshold.

```bash
# search for best threshold on val set (also saves PR curve plot)
python -m stgcn.threshold \
    --checkpoint stgcn/runs/best_model.pt \
    --keypoint_dir gavd_handoff/data/keypoints

# evaluate at fixed threshold on test set
python -m stgcn.threshold \
    --checkpoint stgcn/runs/best_model.pt \
    --keypoint_dir gavd_handoff/data/keypoints \
    --split test --threshold 0.85

# ablation model (note --in_channels 2)
python -m stgcn.threshold \
    --checkpoint stgcn/runs/ablation/best_model.pt \
    --keypoint_dir gavd_handoff/data/keypoints \
    --in_channels 2 --split test --threshold 0.50
```

---

## Results

All results are on the held-out test set (202 sequences, 13 Normal / 189 Abnormal).
AUC is reported on the full probability scores and is threshold-independent.
Macro F1 weights both classes equally and is our primary metric given the class
imbalance.

### Summary

| Model | Threshold | Accuracy | F1 Macro | AUC |
|---|---|---|---|---|
| 1D-CNN Baseline | 0.50 | 0.782 | 0.620 | 0.880 |
| ST-GCN (ours) | 0.50 | 0.847 | 0.663 | 0.899 |
| ST-GCN (ours) | 0.85 (tuned) | 0.876 | 0.661 | 0.899 |
| ST-GCN, C=2 (ablation) | 0.50 | 0.847 | 0.579 | 0.858 |

### Per-class breakdown — ST-GCN at tuned threshold (0.85)

```
              precision    recall  f1-score   support

      Normal      0.286     0.615     0.390        13
    Abnormal      0.971     0.894     0.931       189

    accuracy                          0.876       202
   macro avg      0.628     0.755     0.661       202
weighted avg      0.927     0.876     0.896       202
```

**ST-GCN vs Baseline:** The graph-structured model improves AUC by 1.9 points
(0.880→0.899) and macro F1 by 4.3 points (0.620→0.663), attributable to the
explicit modelling of joint-to-joint relationships via graph convolution.

**Channel ablation:** Removing the confidence channel drops AUC by 4.1 points
(0.899→0.858) and macro F1 by 8.4 points (0.663→0.579). The Normal class is
most affected (F1: 0.42→0.24), suggesting the confidence signal helps the model
identify which joint coordinates are reliable — particularly important for the
minority class where correct detection is harder.

---

## Project Structure

```
.
├── src/                        # Keypoint extraction pipeline
│   ├── extract_gavd.py         # Main extraction script
│   ├── normalize.py            # Hip-centred, torso-scaled normalisation
│   ├── track.py                # Subject selection in multi-person frames
│   ├── skeleton.py             # COCO-17 joint names and edge list
│   └── verify.py               # Overlay visualisation for sanity checking
│
├── baseline/                   # 1D-CNN temporal baseline
│   ├── dataset.py              # Dataset, splits, class weights
│   ├── model.py                # Conv1d encoder with weighted temporal pooling
│   └── train.py                # Training and evaluation script
│
├── stgcn/                      # ST-GCN model
│   ├── graph.py                # Adjacency matrix (3-partition spatial config)
│   ├── model.py                # ST-GCN blocks and full classifier
│   ├── dataset.py              # Dataset (wraps baseline splits)
│   ├── train.py                # Training, checkpointing, and evaluation
│   └── threshold.py            # PR curve + threshold search / test evaluation
│
├── modal_train.py              # Modal GPU training script (upload / train / ablate)
├── gavd_handoff/data/keypoints/ # Extracted .npy, .json, and _labels.csv
└── requirements.txt
```