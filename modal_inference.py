"""
modal_inference.py
------------------
Deploy this file to Modal: `modal deploy modal_inference.py`

It exposes a single remote function `run_gait_inference` that the
Streamlit frontend calls via the Modal Python client.

Architecture:
  Raw video bytes
    → YOLOv8n-Pose  (COCO-17 keypoints, 17×3 per frame)
    → Normalisation (hip-centred, torso-scaled)
    → ST-GCN / 1D-CNN / Ablation model
    → JSON result payload
"""

import modal
import io
import json
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Modal App & Volume
# ─────────────────────────────────────────────────────────────
app   = modal.App("gait-analysis")
vol   = modal.Volume.from_name("gait-model-weights", create_if_missing=True)

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "ultralytics==8.2.0",
        "numpy",
        "opencv-python-headless",
        "scipy",
    )
)

# ─────────────────────────────────────────────────────────────
# COCO-17 skeleton edges
# ─────────────────────────────────────────────────────────────
COCO_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# ─────────────────────────────────────────────────────────────
# Keypoint extraction helper
# ─────────────────────────────────────────────────────────────
def extract_keypoints(video_path: str, yolo_model) -> np.ndarray:
    """
    Run YOLOv8-Pose on every frame and return an array of shape [T, 17, 3].
    Returns None if detection rate < 0.50.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames_kpts = []
    detected = 0
    total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1
        results = yolo_model(frame, verbose=False)
        if results and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            kp = results[0].keypoints.data[0].cpu().numpy()   # [17, 3]
            if kp.shape == (17, 3):
                detected += 1
                frames_kpts.append(kp)
                continue
        # Pad with zeros on detection failure
        frames_kpts.append(np.zeros((17, 3), dtype=np.float32))

    cap.release()

    if total == 0 or (detected / total) < 0.50:
        return None, detected / max(total, 1)

    return np.array(frames_kpts, dtype=np.float32), detected / total   # [T, 17, 3]


def normalise_keypoints(kpts: np.ndarray) -> np.ndarray:
    """
    Hip-centred, torso-scaled normalisation.
    kpts: [T, 17, 3]
    Returns: [T, 17, 3]  (x,y normalised; conf unchanged)
    """
    out = kpts.copy()
    for t in range(len(kpts)):
        # Hip centre
        lhip = kpts[t, 11, :2]
        rhip = kpts[t, 12, :2]
        hip_centre = (lhip + rhip) / 2.0

        # Torso scale: distance between shoulder midpoint and hip midpoint
        lsho = kpts[t, 5, :2]
        rsho = kpts[t, 6, :2]
        sho_centre = (lsho + rsho) / 2.0
        scale = np.linalg.norm(sho_centre - hip_centre) + 1e-6

        out[t, :, :2] = (kpts[t, :, :2] - hip_centre) / scale

    return out


# ─────────────────────────────────────────────────────────────
# Gait feature helpers
# ─────────────────────────────────────────────────────────────
def compute_cadence(kpts: np.ndarray, fps: float = 30.0) -> float:
    """Estimate cadence from vertical ankle oscillation."""
    lankle_y = kpts[:, 15, 1]
    rankle_y = kpts[:, 16, 1]
    combined = (lankle_y + rankle_y) / 2.0

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(combined, distance=fps * 0.3)
    if len(peaks) < 2:
        return 0.0
    duration_s = len(kpts) / fps
    steps = len(peaks) * 2   # each peak = one step
    return (steps / duration_s) * 60.0


def compute_symmetry(kpts: np.ndarray) -> float:
    """
    Symmetry index: 1 - mean absolute diff between left/right limb velocities.
    """
    left_vel  = np.diff(kpts[:, 15, :2], axis=0)   # left ankle
    right_vel = np.diff(kpts[:, 16, :2], axis=0)   # right ankle
    diff = np.abs(np.linalg.norm(left_vel, axis=1) - np.linalg.norm(right_vel, axis=1))
    max_v = np.linalg.norm(right_vel, axis=1).max() + 1e-6
    return float(1.0 - (diff.mean() / max_v))


# ─────────────────────────────────────────────────────────────
# Remote inference function
# ─────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="T4",
    volumes={"/weights": vol},
    timeout=300,
    memory=8192,
)
def run_gait_inference(
    video_bytes:  bytes,
    weight_path:  str,
    model_folder: str,
    in_channels:  int = 3,
    threshold:    float = 0.85,
) -> dict:
    """
    Full inference pipeline.

    Parameters
    ----------
    video_bytes   : Raw MP4 bytes from Streamlit upload.
    weight_path   : Path inside Modal volume, e.g. 'stgcn/runs/best_model.pt'.
    model_folder  : 'stgcn' or 'baseline' — selects which architecture code to import.
    in_channels   : 3 for full pipeline; 2 for ablation (XY only).
    threshold     : Decision boundary for Abnormal.

    Returns
    -------
    JSON-serialisable dict with keys:
      prediction_label, probability_score, prob_timeseries,
      keypoint_tensor, frame_confidences,
      cadence_steps_min, symmetry_index, detection_rate, n_frames
    """
    import torch
    from ultralytics import YOLO

    # ── Write video to temp file ──
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vf:
        vf.write(video_bytes)
        video_path = vf.name

    # ── Load YOLOv8-Pose ──
    yolo = YOLO("yolov8n-pose.pt")   # auto-downloads on first call

    # ── Extract keypoints ──
    kpts_raw, det_rate = extract_keypoints(video_path, yolo)
    os.unlink(video_path)

    if kpts_raw is None:
        return {
            "error": "low_detection",
            "detection_rate": det_rate,
        }

    kpts_norm = normalise_keypoints(kpts_raw)   # [T, 17, 3]

    if in_channels == 2:
        kpts_input = kpts_norm[:, :, :2]        # drop confidence
    else:
        kpts_input = kpts_norm                   # keep confidence

    # ── Load project model ──
    sys.path.insert(0, f"/weights/{model_folder}")
    from model import Model       # noqa
    from graph import Graph       # noqa

    full_weight_path = f"/weights/{weight_path}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    graph = Graph()
    model = Model(
        in_channels=in_channels,
        num_class=2,
        graph=graph,
    ).to(device)
    model.load_state_dict(torch.load(full_weight_path, map_location=device))
    model.eval()

    # ── Prepare tensor  [1, C, T, V, M] ──
    T, V, C = kpts_input.shape
    tensor = torch.from_numpy(kpts_input).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)  # [1,C,T,V,1]
    tensor = tensor.to(device)

    # ── Sliding-window inference for per-frame probabilities ──
    WINDOW = 64
    STRIDE = 8
    prob_series = []

    with torch.no_grad():
        if T >= WINDOW:
            for start in range(0, T - WINDOW + 1, STRIDE):
                seg = tensor[:, :, start:start+WINDOW, :, :]
                out = model(seg)
                p   = torch.softmax(out, dim=1)[0, 1].item()
                prob_series.extend([p] * STRIDE)
            # Trim or pad to T
            prob_series = prob_series[:T]
            if len(prob_series) < T:
                prob_series += [prob_series[-1]] * (T - len(prob_series))
        else:
            # Short sequence: run once
            out = model(tensor)
            p   = torch.softmax(out, dim=1)[0, 1].item()
            prob_series = [p] * T

    # ── Final decision ──
    final_score = float(np.mean(prob_series[-min(10, T):]))
    label       = "Abnormal" if final_score > threshold else "Normal"

    # ── Gait features ──
    cadence  = compute_cadence(kpts_norm)
    symmetry = compute_symmetry(kpts_norm)

    # ── Frame confidences (mean conf per frame) ──
    frame_conf = kpts_raw[:, :, 2].mean(axis=1).tolist()

    return {
        "prediction_label":  label,
        "probability_score": final_score,
        "prob_timeseries":   [float(p) for p in prob_series],
        "keypoint_tensor":   kpts_norm.tolist(),
        "frame_confidences": frame_conf,
        "cadence_steps_min": float(cadence),
        "symmetry_index":    float(symmetry),
        "detection_rate":    float(det_rate),
        "n_frames":          T,
        "model_used":        model_folder,
        "weight_path":       weight_path,
        "in_channels":       in_channels,
    }
