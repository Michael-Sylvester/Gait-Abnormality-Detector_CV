"""
inference.py
------------
Self-contained CPU inference pipeline for GaitVision.
Used as the local fallback when Modal is unavailable, and also
called directly by the Streamlit app when no Modal credentials exist.

Entry point:
    run_inference(video_bytes, weight_path, model_name, in_channels, threshold)
    → dict with the same schema as the Modal remote function
"""

from __future__ import annotations

import os
import sys
import tempfile
import importlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

# ── Ensure project root is on sys.path so stgcn/baseline imports resolve ──
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Google Drive direct-download helper ───────────────────────────────────
GDRIVE_FILE_IDS = {
    # Fill these in once you have the shareable file IDs from your Drive links.
    # Format: "models/<key>": "GDRIVE_FILE_ID"
    # e.g. get the ID from  https://drive.google.com/file/d/<ID>/view
    "stgcn/runs/best_model.pt":              None,  # ← paste ST-GCN file ID
    "stgcn/runs/ablation/best_model.pt":     None,  # ← paste Ablation file ID
    "baseline/runs/best_model.pt":           None,  # ← paste Baseline file ID
}

WEIGHTS_CACHE = PROJECT_ROOT / "model_weights"


def _gdrive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"


def _download_weight(weight_key: str) -> Path:
    """
    Download a model weight from Google Drive if not already cached.
    weight_key: one of the keys in GDRIVE_FILE_IDS, e.g. 'stgcn/runs/best_model.pt'
    Returns local Path to the .pt file.
    """
    local_path = WEIGHTS_CACHE / weight_key
    if local_path.exists():
        return local_path

    file_id = GDRIVE_FILE_IDS.get(weight_key)
    if file_id is None:
        raise FileNotFoundError(
            f"Weight file '{weight_key}' not found locally at {local_path} "
            f"and no Google Drive file ID is configured for it.\n\n"
            f"Either:\n"
            f"  1. Place the .pt file at:  {local_path}\n"
            f"  2. Edit GDRIVE_FILE_IDS in inference.py with the Drive file ID."
        )

    import urllib.request, shutil
    url = _gdrive_url(file_id)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[GaitVision] Downloading {weight_key} from Google Drive…")

    # Google Drive large-file download needs cookie handling
    try:
        import gdown
        gdown.download(url, str(local_path), quiet=False, fuzzy=True)
    except ImportError:
        # Fallback: urllib (may fail for large files without gdown)
        with urllib.request.urlopen(url) as resp, open(local_path, "wb") as f:
            shutil.copyfileobj(resp, f)

    if not local_path.exists() or local_path.stat().st_size < 1000:
        raise RuntimeError(
            f"Downloaded file at {local_path} looks empty. "
            "Install gdown (`pip install gdown`) for reliable large-file downloads."
        )

    print(f"[GaitVision] Saved to {local_path}")
    return local_path


def resolve_weight(weight_key: str) -> Path:
    """
    Resolve a weight path:
      1. Check local repo path  (model_weights/<weight_key>)
      2. Try Google Drive download
    """
    local = WEIGHTS_CACHE / weight_key
    if local.exists():
        return local
    return _download_weight(weight_key)


# ── Model loader (cached in session via module-level dict) ────────────────
_MODEL_CACHE: dict = {}


def load_model(weight_key: str, model_arch: str, in_channels: int):
    """
    Load and cache a model.  model_arch: 'stgcn' | 'baseline'
    """
    cache_key = (weight_key, in_channels)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    weight_path = resolve_weight(weight_key)
    device = torch.device("cpu")

    if model_arch == "stgcn":
        from stgcn.model import STGCN
        model = STGCN(in_channels=in_channels, num_classes=2, dropout=0.0)
    elif model_arch == "baseline":
        from baseline.model import GaitCNN
        # baseline flattens x,y → in_channels is joints*2
        model = GaitCNN(in_channels=17 * 2, num_classes=2, dropout=0.0)
    else:
        raise ValueError(f"Unknown model_arch: {model_arch}")

    state = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    _MODEL_CACHE[cache_key] = model
    return model


# ── Keypoint extraction ───────────────────────────────────────────────────
def extract_keypoints(video_path: str) -> tuple[np.ndarray | None, float]:
    """
    Run YOLOv8n-pose on every frame.
    Returns (kpts [T,17,3], detection_rate) or (None, rate) if rate < 0.50.
    YOLOv8 auto-downloads yolov8n-pose.pt on first call.
    """
    from ultralytics import YOLO

    yolo = YOLO("yolov8n-pose.pt")   # auto-downloads ~6 MB on first use
    cap  = cv2.VideoCapture(video_path)

    frames_kpts, detected, total = [], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1
        results = yolo(frame, verbose=False)
        kp_data = (
            results[0].keypoints.data
            if results and results[0].keypoints is not None
            else None
        )
        if kp_data is not None and len(kp_data) > 0:
            kp = kp_data[0].cpu().numpy()   # [17, 3]  (x, y, conf)
            if kp.shape == (17, 3):
                detected += 1
                frames_kpts.append(kp)
                continue
        frames_kpts.append(np.zeros((17, 3), dtype=np.float32))

    cap.release()
    rate = detected / max(total, 1)
    if total == 0 or rate < 0.50:
        return None, rate
    return np.array(frames_kpts, dtype=np.float32), rate  # [T, 17, 3]


def normalise_keypoints(kpts: np.ndarray) -> np.ndarray:
    """Hip-centred, torso-scaled normalisation. kpts: [T,17,3] → [T,17,3]"""
    out = kpts.copy()
    for t in range(len(kpts)):
        lhip = kpts[t, 11, :2]
        rhip = kpts[t, 12, :2]
        hip_centre = (lhip + rhip) / 2.0

        lsho = kpts[t, 5, :2]
        rsho = kpts[t, 6, :2]
        sho_centre = (lsho + rsho) / 2.0
        scale = max(np.linalg.norm(sho_centre - hip_centre), 1e-6)

        out[t, :, :2] = (kpts[t, :, :2] - hip_centre) / scale
    return out


# ── Gait features ─────────────────────────────────────────────────────────
def compute_cadence(kpts: np.ndarray, fps: float = 30.0) -> float:
    ankle_y = (kpts[:, 15, 1] + kpts[:, 16, 1]) / 2.0
    peaks, _ = find_peaks(ankle_y, distance=max(1, int(fps * 0.3)))
    if len(peaks) < 2:
        return 0.0
    duration_s = len(kpts) / fps
    return (len(peaks) * 2 / duration_s) * 60.0   # steps/min


def compute_symmetry(kpts: np.ndarray) -> float:
    lv = np.diff(kpts[:, 15, :2], axis=0)
    rv = np.diff(kpts[:, 16, :2], axis=0)
    lmag = np.linalg.norm(lv, axis=1)
    rmag = np.linalg.norm(rv, axis=1)
    denom = rmag.max() + 1e-6
    return float(np.clip(1.0 - np.abs(lmag - rmag).mean() / denom, 0.0, 1.0))


# ── Sliding-window inference ───────────────────────────────────────────────
def _run_stgcn_windows(model, kpts_input: np.ndarray, in_channels: int) -> list[float]:
    """kpts_input: [T, 17, C]  → per-frame probability list"""
    device = next(model.parameters()).device
    T = len(kpts_input)
    WINDOW, STRIDE = 64, 8
    prob_series: list[float] = []

    tensor_full = torch.from_numpy(kpts_input).float()  # [T, 17, C]
    # ST-GCN expects (B, C, T, V)
    tensor_full = tensor_full.permute(2, 0, 1).unsqueeze(0)  # [1, C, T, 17]

    with torch.no_grad():
        if T >= WINDOW:
            for start in range(0, T - WINDOW + 1, STRIDE):
                seg  = tensor_full[:, :, start:start + WINDOW, :]
                mask = torch.ones(1, WINDOW, device=device)
                out  = model(seg.to(device), mask)
                p    = torch.softmax(out, dim=1)[0, 1].item()
                prob_series.extend([p] * STRIDE)
            # Pad tail
            prob_series = prob_series[:T]
            if len(prob_series) < T:
                prob_series += [prob_series[-1]] * (T - len(prob_series))
        else:
            # Short clip: run once on full sequence
            mask = torch.ones(1, T, device=device)
            out  = model(tensor_full.to(device), mask)
            p    = torch.softmax(out, dim=1)[0, 1].item()
            prob_series = [p] * T

    return prob_series


def _run_baseline_windows(model, kpts_norm: np.ndarray) -> list[float]:
    """
    Baseline expects (B, 34, T) — x,y only, flattened joints.
    kpts_norm: [T, 17, 3]
    """
    device = next(model.parameters()).device
    T = len(kpts_norm)
    xy = kpts_norm[:, :, :2].reshape(T, -1)  # [T, 34]
    tensor = torch.from_numpy(xy).float().T.unsqueeze(0)  # [1, 34, T]

    with torch.no_grad():
        mask = torch.ones(1, T, device=device)
        out  = model(tensor.to(device), mask)
        p    = torch.softmax(out, dim=1)[0, 1].item()

    return [p] * T


# ── Main entry point ───────────────────────────────────────────────────────
def run_inference(
    video_bytes:  bytes,
    weight_key:   str,
    model_arch:   str,
    in_channels:  int   = 3,
    threshold:    float = 0.85,
) -> dict:
    """
    Full pipeline on CPU.

    Parameters
    ----------
    video_bytes : Raw MP4/AVI bytes from Streamlit uploader.
    weight_key  : e.g. 'stgcn/runs/best_model.pt'
    model_arch  : 'stgcn' | 'baseline'
    in_channels : 3 (x,y,conf) or 2 (x,y only — ablation)
    threshold   : Decision boundary for Abnormal label.

    Returns
    -------
    Dict compatible with the Modal remote function schema.
    """
    # Write video bytes to temp file
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as vf:
        vf.write(video_bytes)
        video_path = vf.name

    try:
        # ── 1. Keypoint extraction ──
        kpts_raw, det_rate = extract_keypoints(video_path)
    finally:
        os.unlink(video_path)

    if kpts_raw is None:
        return {
            "error":          "low_detection",
            "detection_rate": det_rate,
        }

    # ── 2. Normalise ──
    kpts_norm = normalise_keypoints(kpts_raw)   # [T, 17, 3]

    # ── 3. Prepare model input ──
    if in_channels == 2:
        kpts_input = kpts_norm[:, :, :2]        # drop conf channel
    else:
        kpts_input = kpts_norm                  # keep all 3 channels

    # ── 4. Load model ──
    model = load_model(weight_key, model_arch, in_channels)

    # ── 5. Inference ──
    if model_arch == "stgcn":
        prob_series = _run_stgcn_windows(model, kpts_input, in_channels)
    else:
        prob_series = _run_baseline_windows(model, kpts_norm)

    # ── 6. Final decision ──
    T           = len(prob_series)
    tail        = prob_series[-min(10, T):]
    final_score = float(np.mean(tail))
    label       = "Abnormal" if final_score > threshold else "Normal"

    # ── 7. Gait features ──
    cadence  = compute_cadence(kpts_norm)
    symmetry = compute_symmetry(kpts_norm)

    # ── 8. Per-frame mean confidence ──
    frame_conf = kpts_raw[:, :, 2].mean(axis=1).tolist()

    return {
        "prediction_label":  label,
        "probability_score": final_score,
        "prob_timeseries":   [float(p) for p in prob_series],
        "keypoint_tensor":   kpts_norm.tolist(),    # [T, 17, 3]
        "frame_confidences": frame_conf,
        "cadence_steps_min": float(cadence),
        "symmetry_index":    float(symmetry),
        "detection_rate":    float(det_rate),
        "n_frames":          T,
        "model_arch":        model_arch,
        "weight_key":        weight_key,
        "in_channels":       in_channels,
        "_backend":          "cpu",
    }
