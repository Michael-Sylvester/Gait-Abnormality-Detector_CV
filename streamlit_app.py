"""
GaitSense — Gait Abnormality Detection
ICS555: Computer Vision | Ashesi University
AUC: 0.899 | Macro F1: 0.661
"""

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tempfile
import os
import gc
import sys
# Direct imports for models as requested
from stgcn.model import STGCN
from stgcn.graph import build_adjacency
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GaitSense — Gait Abnormality Detection",
    page_icon="🦿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — Medical / Precision Aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

:root {
  --bg:        #07090f;
  --surface:   #0d1117;
  --border:    #1e2530;
  --accent:    #00e5ff;
  --accent2:   #7c3aed;
  --warn:      #f59e0b;
  --danger:    #ef4444;
  --ok:        #10b981;
  --text:      #e2e8f0;
  --muted:     #64748b;
  --card:      #111827;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }

.gs-header {
  padding: 2rem 0 1.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.gs-title {
  font-family: 'Syne', sans-serif;
  font-size: 2.6rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
}
.gs-subtitle {
  font-family: 'DM Mono', monospace;
  font-size: 0.75rem;
  color: var(--muted);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-top: 0.4rem;
}

.verdict-card {
  border-radius: 12px;
  padding: 1.8rem 2rem;
  margin: 1.5rem 0;
  border: 1px solid var(--border);
}
.verdict-normal  { background: linear-gradient(135deg, #052e16 0%, #0d1117 100%); border-color: var(--ok); }
.verdict-abnormal{ background: linear-gradient(135deg, #1c0b0b 0%, #0d1117 100%); border-color: var(--danger); }
.verdict-label   { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; }
.verdict-prob    { font-family: 'DM Mono', monospace; font-size: 0.9rem; color: var(--muted); }

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem;
  margin: 1rem 0;
}
.metric-tile {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.2rem 1rem;
}
.metric-val {
  font-family: 'DM Mono', monospace;
  font-size: 1.6rem;
  font-weight: 500;
  color: var(--accent);
}
.metric-lbl {
  font-size: 0.72rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 0.2rem;
}

.info-tag {
  display: inline-block;
  background: #1e2530;
  color: var(--accent);
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  padding: 3px 10px;
  border-radius: 20px;
  margin: 2px;
  border: 1px solid #2a3545;
}

.section-head {
  font-family: 'Syne', sans-serif;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1rem;
}

.footer-bar {
  margin-top: 3rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  color: var(--muted);
  text-align: center;
}

stButton > button {
  background: var(--accent) !important;
  color: #000 !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: 8px !important;
}

[data-testid="stFileUploader"] {
  border: 2px dashed var(--border) !important;
  border-radius: 12px !important;
  background: var(--surface) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COCO-17 Skeleton Definition
# ─────────────────────────────────────────────
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                    # shoulders
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]
JOINT_NAMES = [
    "Nose","L.Eye","R.Eye","L.Ear","R.Ear",
    "L.Shoulder","R.Shoulder","L.Elbow","R.Elbow",
    "L.Wrist","R.Wrist","L.Hip","R.Hip",
    "L.Knee","R.Knee","L.Ankle","R.Ankle"
]

# ─────────────────────────────────────────────
# Model Loading Utilities (excluding dynamic module loading for STGCN/Graph)
# ─────────────────────────────────────────────
def ensure_gdrive_model(file_id: str, output_path: str):
    """Downloads a model from Google Drive if it doesn't exist locally."""
    if not os.path.exists(output_path) and file_id:
        import gdown
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)

@st.cache_resource(show_spinner=False)
def load_pose_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n-pose.pt")
        model.to("cpu")
        return model
    except Exception as e:
        st.error(f"❌ Could not load YOLOv8 pose model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_classifier(model_choice: str):
    """Load the selected gait classifier."""
    device = torch.device("cpu")
    try:
        if model_choice == "ST-GCN (Primary)":
            ckpt_path  = "stgcn/runs/best_model.pt"
            model_path = "stgcn/model.py"
            graph_path = "stgcn/graph.py"
            
            ensure_gdrive_model("1VMWo1N41NxXkKS9B2eq3KsWrjxa6e-2P", ckpt_path)
            # https://drive.google.com/file/d/1VMWo1N41NxXkKS9B2eq3KsWrjxa6e-2P/view?usp=drive_link
            
            if not (os.path.exists(ckpt_path) and os.path.exists(model_path)):
                return None, "ST-GCN weights not found. Check Google Drive ID or place best_model.pt in gavd-keypoint-extraction-main/stgcn/runs/"
            # STGCN class internally calls build_adjacency
            model      = STGCN(
                in_channels=3, num_classes=2
            ).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            model.eval()
            return model, None

        elif model_choice == "1D-CNN Baseline":
            ckpt_path  = "baseline/runs/best_model.pt"
            model_path = "baseline/model.py"
            
            ensure_gdrive_model("1uujrcmsdYsq91TGnize1NuQdpg1BInX-", ckpt_path)
            # https://drive.google.com/file/d/1uujrcmsdYsq91TGnize1NuQdpg1BInX-/view?usp=drive_link
            
            if not (os.path.exists(ckpt_path) and os.path.exists(model_path)):
                return None, "1D-CNN weights not found. Check Google Drive ID or place best_model.pt in gavd-keypoint-extraction-main/baseline/runs/"
            # Assuming baseline/model.py exists and defines GaitCNN
            from baseline.model import GaitCNN
            model      = GaitCNN().to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            model.eval()
            return model, None

        elif model_choice == "Ablation (XY only)":
            ckpt_path  = "stgcn/runs/ablation/best_model.pt"
            model_path = "stgcn/model.py"
            
            ensure_gdrive_model("1ya3NdQ-iqXmMlhBDt6W_GHLrKTVJ8VLO", ckpt_path)
            # https://drive.google.com/file/d/1ya3NdQ-iqXmMlhBDt6W_GHLrKTVJ8VLO/view?usp=drive_link
            
            if not (os.path.exists(ckpt_path) and os.path.exists(model_path)):
                return None, "Ablation weights not found. Check Google Drive ID or place best_model.pt in gavd-keypoint-extraction-main/stgcn/runs/ablation/"
            # STGCN class internally calls build_adjacency
            model      = STGCN(
                in_channels=2, num_classes=2
            ).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            model.eval()
            return model, None

    except Exception as e:
        return None, f"Model load error: {e}"

    return None, "Unknown model choice"

# ─────────────────────────────────────────────
# Keypoint Normalization
# ─────────────────────────────────────────────
def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    """
    Hip-centered, torso-scaled normalization.
    kpts: (T, 17, 3)  — x, y, conf
    """
    out = kpts.copy().astype(np.float32)
    for t in range(len(out)):
        # Hip center (joints 11 & 12)
        lhip, rhip = out[t, 11, :2], out[t, 12, :2]
        hip_center  = (lhip + rhip) / 2.0

        # Torso scale: distance from hip-center to neck midpoint (5,6)
        lsh, rsh   = out[t, 5, :2], out[t, 6, :2]
        neck        = (lsh + rsh) / 2.0
        scale       = np.linalg.norm(neck - hip_center) + 1e-6

        out[t, :, 0] = (out[t, :, 0] - hip_center[0]) / scale
        out[t, :, 1] = (out[t, :, 1] - hip_center[1]) / scale
    return out

# ─────────────────────────────────────────────
# Video Processing
# ─────────────────────────────────────────────
def process_video(video_path: str, pose_model, progress_bar, status_text, occlusion_vis: bool):
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # We use VP80 / WebM as it provides the most robust native playback in browsers
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out_video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    keypoints_all = []  # (T, 17, 3)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if progress_bar:
            progress_bar.progress(min(frame_idx / max(total, 1), 1.0))
        if status_text:
            status_text.text(f"🔍 Extracting pose — frame {frame_idx}/{total}")

        # YOLOv8 pose inference
        results = pose_model(frame, verbose=False, device="cpu")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        kpts_frame = np.zeros((17, 3), dtype=np.float32)
        if results and results[0].keypoints is not None:
            kp = results[0].keypoints
            if kp.data.shape[0] > 0:
                kp_np = kp.data[0].cpu().numpy()   # (17, 3)
                kpts_frame = kp_np[:17]

                # Draw skeleton
                        frame_rgb = draw_skeleton(frame_rgb, kpts_frame, occlusion_mode=occlusion_vis)

        out_video.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        keypoints_all.append(kpts_frame)

    cap.release()
    out_video.release()
    return out_path, np.array(keypoints_all), fps

def draw_skeleton(img: np.ndarray, kpts: np.ndarray,
                  occlusion_mode=False, conf_threshold=0.3) -> np.ndarray:
    """Draw COCO-17 skeleton on an RGB image."""
    h, w = img.shape[:2]
    out  = img.copy()

    # Draw edges
    for (a, b) in COCO_EDGES:
        if kpts[a, 2] > conf_threshold and kpts[b, 2] > conf_threshold:
            xa, ya = int(kpts[a, 0] * w) if kpts[a, 0] <= 1 else int(kpts[a, 0]), \
                     int(kpts[a, 1] * h) if kpts[a, 1] <= 1 else int(kpts[a, 1])
            xb, yb = int(kpts[b, 0] * w) if kpts[b, 0] <= 1 else int(kpts[b, 0]), \
                     int(kpts[b, 1] * h) if kpts[b, 1] <= 1 else int(kpts[b, 1])
            color = (0, 229, 255)
            cv2.line(out, (xa, ya), (xb, yb), color, 2, cv2.LINE_AA)

    # Draw joints
    for j in range(17):
        conf = kpts[j, 2]
        if conf > conf_threshold:
            x = int(kpts[j, 0] * w) if kpts[j, 0] <= 1 else int(kpts[j, 0])
            y = int(kpts[j, 1] * h) if kpts[j, 1] <= 1 else int(kpts[j, 1])
            if occlusion_mode:
                # Color by confidence: red (low) → green (high)
                r = int(255 * (1 - conf))
                g = int(255 * conf)
                color = (r, g, 30)
            else:
                color = (124, 58, 237) if j in (11, 12) else (0, 229, 255)
            cv2.circle(out, (x, y), 5, color, -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), 7, (255, 255, 255), 1, cv2.LINE_AA)

    return out

# ─────────────────────────────────────────────
# Gait Metrics
# ─────────────────────────────────────────────
def compute_gait_metrics(kpts: np.ndarray, fps: float) -> dict:
    """
    kpts: (T, 17, 3) — raw pixel coords
    Returns cadence (steps/min) and step symmetry ratio.
    """
    T = len(kpts)
    if T < 4:
        return {"cadence": "N/A", "step_symmetry": "N/A",
                "avg_confidence": "N/A", "n_frames": T}

    # Use ankle vertical motion as step proxy
    l_ankle_y = kpts[:, 15, 1]
    r_ankle_y = kpts[:, 16, 1]
    l_conf    = kpts[:, 15, 2].mean()
    r_conf    = kpts[:, 16, 2].mean()

    def count_steps(signal):
        """Count peaks in a 1-D signal (simple threshold crossing)."""
        if signal.std() < 1:
            return 0
        sig_norm = (signal - signal.mean()) / (signal.std() + 1e-6)
        peaks = 0
        for i in range(1, len(sig_norm)-1):
            if sig_norm[i] > 0.5 and sig_norm[i] > sig_norm[i-1] and sig_norm[i] > sig_norm[i+1]:
                peaks += 1
        return peaks

    l_steps = count_steps(l_ankle_y) if l_conf > 0.3 else 0
    r_steps = count_steps(r_ankle_y) if r_conf > 0.3 else 0
    total_steps  = l_steps + r_steps
    duration_min = T / fps / 60
    cadence      = round(total_steps / (duration_min + 1e-6))
    cadence      = min(cadence, 200)   # sanity cap

    sym_ratio = "N/A"
    if l_steps > 0 and r_steps > 0:
        sym_ratio = f"{min(l_steps, r_steps) / max(l_steps, r_steps):.3f}"

    avg_conf = kpts[:, :, 2].mean()

    return {
        "cadence":         f"{cadence} steps/min",
        "step_symmetry":   sym_ratio,
        "avg_confidence":  f"{avg_conf:.2f}",
        "n_frames":        T,
    }

# ─────────────────────────────────────────────
# Classifier Inference
# ─────────────────────────────────────────────
THRESHOLD = 0.85

def run_inference(model, kpts_norm: np.ndarray, model_choice: str, status_text):
    """
    kpts_norm: (T, 17, 3) normalized
    Returns: prob_abnormal (float), temporal_probs (list[float])
    """
    device  = torch.device("cpu")
    T       = kpts_norm.shape[0]
    # 1D-CNN Baseline must use 2 channels (XY) to meet its 34-channel expectation
    # 1D-CNN Baseline must use 2 channels (XY) to meet its 34-channel expectation
    in_ch   = 2 if ("Ablation" in model_choice or "1D-CNN" in model_choice) else 3

    # Build full tensor: (C, T, 17)
    seq = kpts_norm[:, :, :in_ch]   # (T, 17, C)
    seq = seq.transpose(2, 0, 1)    # (C, T, 17)
    x   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, C, T, 17)

    # Create mask: (1, T) representing per-frame mean confidence
    frame_conf = kpts_norm[:, :, 2].mean(axis=1) # (T,)
    mask = torch.tensor(frame_conf, dtype=torch.float32).unsqueeze(0).to(device) # (1, T)

    # Sliding window for temporal probabilities
    WIN    = min(64, T)
    STRIDE = max(1, WIN // 4)
    probs_t = []

    with torch.no_grad():
        if status_text:
            status_text.text("🧠 Running graph neural network inference …")

        if model_choice == "1D-CNN Baseline":
            # 1D-CNN expects (B, C*17, T)
            flat = x.reshape(1, in_ch * 17, T)
            logits = model(flat, mask)
        else:
            # ST-GCN expects (B, C, T, 17)
            logits = model(x, mask)

        prob = F.softmax(logits, dim=1)[0, 1].item()

        # Temporal sliding window
        for start in range(0, T - WIN + 1, STRIDE):
            chunk = seq[:, start:start+WIN, :]
            cx    = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            chunk_mask = mask[:, start:start+WIN]
            with torch.no_grad():
                if model_choice == "1D-CNN Baseline":
                    flat2  = cx.reshape(1, in_ch * 17, WIN)
                    logits2 = model(flat2, chunk_mask)
                else:
                    logits2 = model(cx, chunk_mask)
                p2 = F.softmax(logits2, dim=1)[0, 1].item()
            probs_t.append((start + WIN // 2, p2))

    return prob, probs_t

# ─────────────────────────────────────────────
# Plotly Charts
# ─────────────────────────────────────────────
def make_temporal_chart(probs_t: list, fps: float):
    if not probs_t:
        return None
    frames_x = [p[0] / fps for p in probs_t]
    probs_y  = [p[1] for p in probs_t]

    fig = go.Figure()
    fig.add_hrect(y0=THRESHOLD, y1=1.0,
                  fillcolor="rgba(239,68,68,0.07)", line_width=0)
    fig.add_hline(y=THRESHOLD, line_dash="dash",
                  line_color="#ef4444", line_width=1.5,
                  annotation_text=f"  Threshold ({THRESHOLD})",
                  annotation_font_color="#ef4444",
                  annotation_font_size=11)
    # Area
    fig.add_trace(go.Scatter(
        x=frames_x + frames_x[::-1],
        y=probs_y + [0]*len(probs_y),
        fill="toself",
        fillcolor="rgba(0,229,255,0.08)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=frames_x, y=probs_y,
        mode="lines+markers",
        name="P(Abnormal)",
        line=dict(color="#00e5ff", width=2.5),
        marker=dict(size=6, color="#00e5ff",
                    line=dict(color="#07090f", width=1.5)),
        hovertemplate="<b>t=%.2fs</b><br>P(Abnormal): %.3f<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(family="DM Mono, monospace", size=11, color="#94a3b8"),
        xaxis=dict(title="Time (s)", gridcolor="#1e2530", showline=False, zeroline=False),
        yaxis=dict(title="P(Abnormal)", range=[0,1], gridcolor="#1e2530",
                   showline=False, zeroline=False),
        margin=dict(l=50, r=20, t=20, b=50),
        showlegend=False,
        height=280,
    )
    return fig

def make_confidence_chart(kpts: np.ndarray):
    """Bar chart of per-joint confidence averages."""
    avg_conf = kpts[:, :, 2].mean(axis=0)  # (17,)
    colors   = ["#00e5ff" if c > 0.5 else "#f59e0b" if c > 0.3 else "#ef4444"
                for c in avg_conf]
    fig = go.Figure(go.Bar(
        x=JOINT_NAMES, y=avg_conf,
        marker_color=colors,
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(family="DM Mono, monospace", size=10, color="#94a3b8"),
        xaxis=dict(tickangle=-45, gridcolor="#1e2530"),
        yaxis=dict(title="Avg Confidence", range=[0, 1], gridcolor="#1e2530"),
        margin=dict(l=40, r=10, t=20, b=80),
        height=260,
    )
    return fig

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
      <div style='font-family: Syne, sans-serif; font-size:1.4rem; font-weight:800;
                  background: linear-gradient(135deg,#00e5ff,#7c3aed);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        GaitSense
      </div>
      <div style='font-family:DM Mono,monospace; font-size:0.65rem; color:#475569;
                  letter-spacing:0.12em; text-transform:uppercase; margin-top:0.2rem;'>
        ICS555 · Ashesi University
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Model Selection</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Classifier",
        ["ST-GCN (Primary)", "1D-CNN Baseline", "Ablation (XY only)"],
        index=0,
        help="ST-GCN uses graph convolutions to model spatial joint relationships — key advantage over 1D-CNN baseline."
    )

    if model_choice == "ST-GCN (Primary)":
        st.markdown("""
        <div style='background:#0d1117;border:1px solid #1e2530;border-radius:8px;
                    padding:0.9rem;margin-top:0.5rem;font-size:0.75rem;color:#64748b;
                    font-family:Inter,sans-serif;'>
          <span style='color:#00e5ff;font-weight:600;'>Graph Advantage:</span> ST-GCN 
          encodes <em>joint-to-joint spatial dependencies</em> via an adjacency matrix, 
          capturing how knee flexion correlates with hip rotation — impossible with 1D-CNN.
        </div>
        """, unsafe_allow_html=True)
    elif model_choice == "Ablation (XY only)":
        st.markdown("""
        <div style='background:#0d1117;border:1px solid #f59e0b44;border-radius:8px;
                    padding:0.9rem;margin-top:0.5rem;font-size:0.75rem;color:#64748b;'>
          <span style='color:#f59e0b;font-weight:600;'>Ablation Study:</span> 
          Confidence channel removed (in_channels=2). Occlusion Visualizer highlights 
          what the full model gains by weighting uncertain joints.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-head" style="margin-top:1.5rem;">Visualization</div>',
                unsafe_allow_html=True)
    occlusion_vis = st.toggle("Occlusion Visualizer",
                              value=(model_choice == "Ablation (XY only)"),
                              help="Color joints by confidence: green=high, red=low")
    show_conf_chart = st.toggle("Joint Confidence Chart", value=True)

    st.markdown('<div class="section-head" style="margin-top:1.5rem;">Performance</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-tile" style="margin-bottom:0.5rem;">
      <div class="metric-val">0.899</div>
      <div class="metric-lbl">AUC — ST-GCN</div>
    </div>
    <div class="metric-tile">
      <div class="metric-val">0.661</div>
      <div class="metric-lbl">Macro F1 — ST-GCN</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="gs-header">
  <div class="gs-title">GaitSense</div>
  <div class="gs-subtitle">Spatiotemporal Graph Convolutional Gait Analysis · GAVD Dataset · 1,874 Sequences</div>
  <div style="margin-top:0.8rem;">
    <span class="info-tag">YOLOv8-Nano Pose</span>
    <span class="info-tag">ST-GCN</span>
    <span class="info-tag">COCO-17 Keypoints</span>
    <span class="info-tag">CPU Inference</span>
    <span class="info-tag">Threshold τ=0.85</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# File Upload
# ─────────────────────────────────────────────
st.markdown('<div class="section-head">Upload Gait Video</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "MP4 / AVI / MOV (single subject, frontal or sagittal view)",
    type=["mp4", "avi", "mov", "mkv"],
    label_visibility="collapsed"
)

# ─────────────────────────────────────────────
# Main Processing
# ─────────────────────────────────────────────
if uploaded:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-head">Pose Extraction · Skeleton Overlay</div>',
                    unsafe_allow_html=True)
        video_placeholder = st.empty()
        progress_bar = st.progress(0.0)
        status_text  = st.empty()

    with col_right:
        st.markdown('<div class="section-head">Abnormality Probability · Temporal</div>',
                    unsafe_allow_html=True)
        chart_placeholder = st.empty()

    # ── Load models
    with st.spinner("Loading pose model …"):
        pose_model = load_pose_model()

    if pose_model is None:
        st.error("YOLOv8 pose model unavailable. Install ultralytics: `pip install ultralytics`")
        st.stop()

    with st.spinner(f"Loading {model_choice} …"):
        classifier, err_msg = load_classifier(model_choice)

    if err_msg:
        st.warning(f"⚠️ {err_msg}  \nRunning **pose-only** mode (no classification).")
        classifier = None

    # ── Process video
    out_video_path, kpts_raw, fps = process_video(
        tmp_path, pose_model, progress_bar, status_text, occlusion_vis
    )
    progress_bar.empty()

    # Normalize keypoints
    kpts_norm = normalize_keypoints(kpts_raw)

    # ── Display reconstructed video
    with col_left:
        with open(out_video_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        if occlusion_vis:
            st.markdown("""
            <div style='font-family:DM Mono,monospace;font-size:0.72rem;color:#64748b;
                        padding:0.5rem 0;'>
              🟢 <b style='color:#10b981'>High confidence</b> &nbsp;|&nbsp; 
              🔴 <b style='color:#ef4444'>Low confidence / occluded</b>
            </div>""", unsafe_allow_html=True)

    # ── Inference
    prob_abnormal = None
    probs_t       = []
    if classifier is not None:
        status_text.text("🧠 Running classifier …")
        try:
            prob_abnormal, probs_t = run_inference(
                classifier, kpts_norm, model_choice, status_text
            )
        except Exception as e:
            st.error(f"Inference error: {e}")
        status_text.empty()

    # ── Temporal chart
    with col_right:
        if probs_t:
            fig_t = make_temporal_chart(probs_t, fps)
            chart_placeholder.plotly_chart(fig_t, use_container_width=True)
        else:
            chart_placeholder.info("Classification model not loaded — no probability curve.")

    # ── Verdict
    if prob_abnormal is not None:
        is_abnormal = prob_abnormal >= THRESHOLD
        if is_abnormal:
            st.markdown(f"""
            <div class="verdict-card verdict-abnormal">
              <div class="verdict-label" style="color:#ef4444;">⚠ ABNORMAL GAIT DETECTED</div>
              <div class="verdict-prob" style="margin-top:0.4rem;">
                P(Abnormal) = <b style="color:#ef4444;">{prob_abnormal:.4f}</b> &nbsp;≥&nbsp; τ = {THRESHOLD}
                &nbsp;|&nbsp; Model: {model_choice}
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-card verdict-normal">
              <div class="verdict-label" style="color:#10b981;">✓ NORMAL GAIT</div>
              <div class="verdict-prob" style="margin-top:0.4rem;">
                P(Abnormal) = <b style="color:#10b981;">{prob_abnormal:.4f}</b> &nbsp;&lt;&nbsp; τ = {THRESHOLD}
                &nbsp;|&nbsp; Model: {model_choice}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Gait Metrics
    metrics = compute_gait_metrics(kpts_raw, fps)
    st.markdown('<div class="section-head" style="margin-top:1.5rem;">Technical Gait Metrics</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-tile">
        <div class="metric-val">{metrics['cadence']}</div>
        <div class="metric-lbl">Cadence</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{metrics['step_symmetry']}</div>
        <div class="metric-lbl">Step Symmetry Index</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{metrics['avg_confidence']}</div>
        <div class="metric-lbl">Avg Keypoint Confidence</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{metrics['n_frames']}</div>
        <div class="metric-lbl">Frames Processed</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Joint Confidence Chart
    if show_conf_chart:
        st.markdown('<div class="section-head" style="margin-top:1.5rem;">Joint Confidence Profile</div>',
                    unsafe_allow_html=True)
        fig_c = make_confidence_chart(kpts_raw)
        st.plotly_chart(fig_c, use_container_width=True)

    # ── Memory cleanup
    gc.collect()
    os.unlink(tmp_path)
    if os.path.exists(out_video_path):
        os.unlink(out_video_path)

else:
    # Landing state
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem; border: 1px dashed #1e2530;
                border-radius: 16px; background: #0d1117; margin: 2rem 0;'>
      <div style='font-size: 3rem; margin-bottom: 1rem;'>🦿</div>
      <div style='font-family: Syne, sans-serif; font-size: 1.4rem; font-weight: 700;
                  color: #e2e8f0; margin-bottom: 0.6rem;'>
        Upload a gait video to begin analysis
      </div>
      <div style='font-family: DM Mono, monospace; font-size: 0.8rem; color: #475569;
                  max-width: 480px; margin: 0 auto; line-height: 1.7;'>
        Accepts MP4 · AVI · MOV · MKV<br>
        Best results: frontal or sagittal view, single subject, stable camera
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Architecture diagram description
    st.markdown('<div class="section-head">Pipeline Architecture</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4, gap="small")
    steps = [
        ("01", "Video Input", "Frame-by-frame OpenCV decoding"),
        ("02", "YOLOv8-Nano Pose", "COCO-17 keypoint extraction per frame"),
        ("03", "Normalization", "Hip-centered · Torso-scaled"),
        ("04", "ST-GCN", "Spatiotemporal graph convolution → P(Abnormal)"),
    ]
    for col, (num, title, desc) in zip([col1, col2, col3, col4], steps):
        with col:
            st.markdown(f"""
            <div class="metric-tile" style="text-align:center; padding: 1.4rem 1rem;">
              <div style='font-family:DM Mono,monospace; font-size:0.7rem; color:#475569;
                          margin-bottom:0.4rem;'>{num}</div>
              <div style='font-family:Syne,sans-serif; font-weight:700; font-size:0.9rem;
                          color:#00e5ff; margin-bottom:0.4rem;'>{title}</div>
              <div style='font-size:0.73rem; color:#64748b; font-family:Inter,sans-serif;
                          line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
  ICS555: Computer Vision &nbsp;·&nbsp; Ashesi University &nbsp;·&nbsp;
  GAVD Dataset (1,874 sequences) &nbsp;·&nbsp;
  <b style="color:#00e5ff;">AUC 0.899</b> &nbsp;·&nbsp;
  <b style="color:#7c3aed;">Macro F1 0.661</b> &nbsp;·&nbsp;
  YOLOv8-Nano + ST-GCN · CPU Inference
</div>
""", unsafe_allow_html=True)
