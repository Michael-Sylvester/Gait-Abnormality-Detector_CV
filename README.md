# GaitVision — ST-GCN Gait Analysis
**ICS555: Computer Vision · Ashesi University**

> Classify walking videos as **Normal** or **Abnormal** using  
> YOLOv8-Pose → Spatio-Temporal Graph Convolutional Networks (ST-GCN)  
> on the GAVD dataset (1,874 sequences).

---

## Architecture

```
Raw Video
  └─► YOLOv8n-Pose  ──► COCO-17 keypoints [T × 17 × 3]
        └─► Hip-centred normalisation
              └─► ST-GCN / 1D-CNN / Ablation model
                    └─► Normal / Abnormal + per-frame probabilities
```

**Model Results (GAVD Test Set)**

| Model | AUC-ROC | Macro F1 |
|---|---|---|
| ST-GCN (Main) | **0.899** | **0.661** |
| 1D-CNN Baseline | 0.761 | 0.512 |
| Channel Ablation (XY only) | 0.843 | 0.598 |

---

## Deployment

### 1 — Backend (your teammate's action)

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new

# Deploy the inference endpoint
modal deploy modal_inference.py
```

The Modal app will be reachable as `gait-analysis`.  
Your teammate should share their `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` with you.

### 2 — Frontend (Streamlit Cloud)

1. Push this repo to GitHub.
2. Connect the repo at [share.streamlit.io](https://share.streamlit.io).
3. Set the main file to `app.py`.
4. In **Settings → Secrets**, add:

```toml
MODAL_TOKEN_ID     = "ak-..."
MODAL_TOKEN_SECRET = "as-..."
```

### 3 — Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enable **Demo Mode** in the sidebar to test without Modal credentials.

---

## Project Structure

```
.
├── app.py                  # Streamlit frontend (main entry point)
├── modal_inference.py      # Modal backend (deploy with `modal deploy`)
├── src/
│   └── skeleton.py         # COCO-17 edges + adjacency matrix builder
├── stgcn/
│   ├── model.py            # ST-GCN architecture
│   ├── graph.py            # Graph topology helper
│   └── runs/
│       ├── best_model.pt
│       └── ablation/
│           └── best_model.pt
├── baseline/
│   ├── model.py            # 1D-CNN architecture
│   └── runs/
│       └── best_model.pt
├── requirements.txt
└── .streamlit/
    ├── config.toml         # Dark medical theme
    └── secrets.toml        # ← DO NOT COMMIT WITH REAL KEYS
```

---

## Presentation Tips (Technical Depth — 60% of grade)

1. **Live model switching**: Switch from ST-GCN → 1D-CNN during the demo to show the  
   AUC gap (0.899 vs 0.761) in real time on the same video.
2. **Occlusion Sensitivity toggle**: Enables the confidence-channel colour overlay,  
   directly demonstrating the ablation study finding that Channel 3 (confidence) provides  
   meaningful signal beyond pure geometry.
3. **Probability timeseries chart**: Point to the frame-level fluctuations to explain  
   the sliding-window inference strategy.

---

## Citations

- **GAVD Dataset**: MV-TGCN — https://github.com/niais/mv-tgcn
- **ST-GCN**: Yan et al. (2018) — *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*
- **YOLOv8-Pose**: Ultralytics — https://docs.ultralytics.com/tasks/pose/
- **Modal Serverless Inference**: https://modal.com
