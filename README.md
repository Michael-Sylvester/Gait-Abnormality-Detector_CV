# GaitSense — Gait Abnormality Detection
**ICS555: Computer Vision · Ashesi University**

> ST-GCN-powered gait classification on the GAVD dataset.  
> **AUC: 0.899 · Macro F1: 0.661**

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected Directory Layout

```
.
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── stgcn/
│   ├── model.py          # ST_GCN class
│   ├── graph.py          # Graph class
│   └── runs/
│       ├── best_model.pt          # Primary ST-GCN weights
│       └── ablation/
│           └── best_model.pt      # Ablation (XY only, in_channels=2)
└── baseline/
    ├── model.py          # GaitCNN class
    └── runs/
        └── best_model.pt          # 1D-CNN baseline weights
```

## Model Requirements

### ST-GCN (`stgcn/model.py`)
Must expose `STGCN(in_channels, num_classes, dropout)`.

### Graph (`stgcn/graph.py`)
Must expose `build_adjacency(num_joints, edges, root)`.

### 1D-CNN Baseline (`baseline/model.py`)
Must expose `GaitCNN()` accepting input shape `(B, C, T*17)`.

## Pipeline

```
Video → YOLOv8n-pose (CPU) → COCO-17 keypoints (T×17×3)
      → Hip-center + torso-scale normalization
      → Tensor (1, C, T, 17, 1)
      → ST-GCN forward pass
      → Softmax → P(Abnormal) vs threshold τ=0.85
```

## UI Features

| Feature | Description |
|---|---|
| Dual-view dashboard | Skeleton overlay + temporal probability chart |
| Model switcher | ST-GCN / 1D-CNN / Ablation via sidebar |
| Occlusion Visualizer | Joint color by confidence (red→green) |
| Gait Metrics | Cadence, Step Symmetry Index |
| Joint Confidence Chart | Per-joint average confidence bar chart |

## Deployment Notes

- All inference runs on **CPU** — no GPU required.
- Memory: frames are cleared after processing to stay under Streamlit Cloud's ~1 GB limit.
- Upload limit: 200 MB (set in `.streamlit/config.toml`).
- `opencv-python-headless` is used (no display server needed).

---
*GAVD Dataset · 1,874 sequences · Normal / Abnormal classification*
