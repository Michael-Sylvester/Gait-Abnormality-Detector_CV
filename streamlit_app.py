"""
GaitVision — ST-GCN Gait Analysis Dashboard
ICS555: Computer Vision | Ashesi University
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import tempfile
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GaitVision · Ashesi CV",
    page_icon="🦿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Medical / Clinical dark theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

/* ── Root tokens ── */
:root {
  --bg-base:    #050d1a;
  --bg-card:    #0a1628;
  --bg-panel:   #0d1e36;
  --accent:     #00d4ff;
  --accent2:    #00ffc2;
  --danger:     #ff4b6e;
  --warning:    #ffb347;
  --text-hi:    #e8f4fd;
  --text-mid:   #7fa8c9;
  --text-lo:    #3a5a78;
  --border:     #1a3050;
  --radius:     8px;
  --glow:       0 0 20px rgba(0,212,255,.18);
}

/* ── Global reset ── */
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg-base) !important; color: var(--text-hi); }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-card) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-hi) !important; }
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSlider label { font-size: 0.82rem; color: var(--text-mid) !important; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

/* ── Cards ── */
.gv-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
  box-shadow: var(--glow);
}
.gv-card-title {
  font-family: 'DM Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: .6rem;
}

/* ── Metric chips ── */
.metric-row { display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0; }
.metric-chip {
  flex:1; min-width:110px;
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: .8rem 1rem;
  text-align: center;
}
.metric-chip .val {
  font-family: 'Syne', sans-serif;
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--accent);
  line-height:1;
}
.metric-chip .lbl {
  font-size: 0.7rem;
  color: var(--text-mid);
  margin-top: .25rem;
  font-family: 'DM Mono', monospace;
  letter-spacing:.06em;
}

/* ── Status badge ── */
.badge {
  display: inline-block;
  padding: .25rem .75rem;
  border-radius: 4px;
  font-family: 'DM Mono', monospace;
  font-size: .75rem;
  letter-spacing: .08em;
  font-weight: 500;
}
.badge-normal  { background: rgba(0,255,194,.12); color: var(--accent2); border:1px solid rgba(0,255,194,.3); }
.badge-abnormal{ background: rgba(255,75,110,.12); color: var(--danger);  border:1px solid rgba(255,75,110,.3); }
.badge-info    { background: rgba(0,212,255,.10); color: var(--accent);  border:1px solid rgba(0,212,255,.25); }

/* ── Probability bar ── */
.prob-bar-wrap { height:8px; background:var(--bg-panel); border-radius:4px; margin:.5rem 0; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:4px; transition:width .6s ease; }

/* ── Header ── */
.gv-header {
  display:flex; align-items:baseline; gap:1rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 1rem;
  margin-bottom: 1.5rem;
}
.gv-header h1 { font-size:1.9rem; font-weight:800; color:var(--text-hi); margin:0; }
.gv-header .sub {
  font-family:'DM Mono',monospace;
  font-size:.72rem;
  color:var(--text-mid);
  letter-spacing:.1em;
  text-transform:uppercase;
}

/* ── Footer ── */
.gv-footer {
  margin-top:3rem;
  padding-top:1.25rem;
  border-top:1px solid var(--border);
  font-size:.75rem;
  color:var(--text-lo);
  font-family:'DM Mono',monospace;
  display:flex; gap:2rem; flex-wrap:wrap; align-items:center;
}
.gv-footer a { color:var(--text-mid); text-decoration:none; }
.gv-footer a:hover { color:var(--accent); }
.stat-pill {
  display:inline-flex; gap:.4rem; align-items:center;
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:4px;
  padding:.2rem .6rem;
}
.stat-pill span { color:var(--accent); font-weight:500; }

/* ── Plotly container ── */
.js-plotly-plot { border-radius: var(--radius); }

/* ── Buttons ── */
.stButton>button {
  background: linear-gradient(135deg, #00d4ff22, #00ffc211) !important;
  border: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  border-radius: 6px !important;
  font-family: 'DM Mono', monospace !important;
  letter-spacing: .06em !important;
  font-size: .82rem !important;
  transition: all .2s !important;
}
.stButton>button:hover {
  background: linear-gradient(135deg, #00d4ff44, #00ffc233) !important;
  box-shadow: var(--glow) !important;
}

/* ── Radio ── */
.stRadio [data-baseweb="radio"] { accent-color: var(--accent); }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
  border: 1px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--bg-panel) !important;
}

/* ── Toggle ── */
.stToggle [data-baseweb="checkbox"] { accent-color: var(--accent); }

/* ── Info/Warning boxes ── */
.stAlert { border-radius: var(--radius) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg-base); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }

/* ── Skeleton overlay container ── */
.skeleton-container {
  position:relative;
  border-radius: var(--radius);
  overflow:hidden;
  background: #000;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# COCO-17 SKELETON  (mirrors src/skeleton.py)
# ─────────────────────────────────────────────────────────────
COCO_EDGES = [
    (0,1),(0,2),(1,3),(2,4),          # head
    (5,6),(5,7),(7,9),(6,8),(8,10),   # arms
    (5,11),(6,12),(11,12),            # torso
    (11,13),(13,15),(12,14),(14,16),  # legs
]
COCO_KP_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]
LEFT_KPS  = [1,3,5,7,9,11,13,15]
RIGHT_KPS = [2,4,6,8,10,12,14,16]

# ─────────────────────────────────────────────────────────────
# MODEL CONFIGS
# ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "ST-GCN (Main)": {
        "weight_path":  "stgcn/runs/best_model.pt",
        "model_folder": "stgcn",
        "in_channels":  3,
        "description":  "Spatio-Temporal GCN on COCO-17 skeleton sequences",
        "default_thresh": 0.85,
    },
    "1D-CNN Baseline": {
        "weight_path":  "baseline/runs/best_model.pt",
        "model_folder": "baseline",
        "in_channels":  3,
        "description":  "1-D CNN baseline over flattened keypoint timeseries",
        "default_thresh": 0.50,
    },
    "Channel Ablation": {
        "weight_path":  "stgcn/runs/ablation/best_model.pt",
        "model_folder": "stgcn",
        "in_channels":  2,          # excludes confidence channel
        "description":  "ST-GCN ablation: XY only (confidence excluded)",
        "default_thresh": 0.80,
    },
}

# ─────────────────────────────────────────────────────────────
# MODAL CLIENT HELPER
# ─────────────────────────────────────────────────────────────
def get_modal_client():
    """Return a connected Modal app, or None on failure."""
    try:
        import modal
        token_id     = st.secrets["MODAL_TOKEN_ID"]
        token_secret = st.secrets["MODAL_TOKEN_SECRET"]
        os.environ["MODAL_TOKEN_ID"]     = token_id
        os.environ["MODAL_TOKEN_SECRET"] = token_secret
        app = modal.App.lookup("gait-analysis")
        return app
    except Exception as e:
        return None


def call_modal_inference(video_bytes: bytes, model_name: str, threshold: float) -> dict:
    """
    Call the remote Modal inference function.
    Returns a standardised result dict.
    """
    cfg = MODEL_CONFIGS[model_name]
    app = get_modal_client()
    if app is None:
        raise RuntimeError(
            "Could not connect to Modal. "
            "Check that MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are set "
            "in Streamlit Secrets and the Modal app 'gait-analysis' is deployed."
        )
    try:
        run_inference = app.function("run_gait_inference")
        result = run_inference.remote(
            video_bytes   = video_bytes,
            weight_path   = cfg["weight_path"],
            model_folder  = cfg["model_folder"],
            in_channels   = cfg["in_channels"],
            threshold     = threshold,
        )
        return result
    except Exception as e:
        raise RuntimeError(f"Modal inference failed: {e}")


# ─────────────────────────────────────────────────────────────
# MOCK INFERENCE  (demo / no Modal credentials)
# ─────────────────────────────────────────────────────────────
def mock_inference(model_name: str, n_frames: int = 60) -> dict:
    """Generate plausible mock results for UI demonstration."""
    rng = np.random.default_rng(42)
    is_abnormal = model_name != "1D-CNN Baseline"

    # Probability timeseries
    base = 0.78 if is_abnormal else 0.22
    noise = rng.normal(0, 0.06, n_frames)
    probs = np.clip(base + noise, 0, 1).tolist()

    # Final score
    score = float(np.mean(probs[-10:]))
    label = "Abnormal" if score > 0.5 else "Normal"

    # Fake keypoints  [T, 17, 3]  (x, y, conf)
    t = np.linspace(0, 2*np.pi, n_frames)
    kpts = []
    for i in range(n_frames):
        frame_kpts = []
        for j in range(17):
            x    = 0.3 + 0.4 * (j % 4) / 3 + 0.02*np.sin(t[i]+j)
            y    = 0.1 + 0.8 * (j // 4) / 4 + 0.015*np.cos(t[i]+j*0.7)
            conf = float(rng.uniform(0.6, 0.99))
            frame_kpts.append([x, y, conf])
        kpts.append(frame_kpts)

    # Frame confidences (mean per frame)
    frame_conf = [float(np.mean([k[2] for k in f])) for f in kpts]

    # Gait features
    cadence = rng.uniform(90, 115)
    symmetry = rng.uniform(0.82, 0.97) if label == "Normal" else rng.uniform(0.55, 0.78)

    return {
        "prediction_label":  label,
        "probability_score": score,
        "prob_timeseries":   probs,
        "keypoint_tensor":   kpts,
        "frame_confidences": frame_conf,
        "cadence_steps_min": float(cadence),
        "symmetry_index":    float(symmetry),
        "detection_rate":    float(rng.uniform(0.82, 0.99)),
        "n_frames":          n_frames,
        "model_used":        model_name,
    }


# ─────────────────────────────────────────────────────────────
# SKELETON SVG RENDERER
# ─────────────────────────────────────────────────────────────
def render_skeleton_svg(kpts_frame: list, occlusion: bool,
                        width: int = 400, height: int = 500) -> str:
    """
    Render a single frame's skeleton as an inline SVG.
    kpts_frame: list of [x_norm, y_norm, conf]  (17 points, normalised 0-1)
    """
    def c2s(x_n, y_n):
        return x_n * width, y_n * height

    def conf_color(conf: float) -> str:
        if not occlusion:
            return "#00d4ff"
        r = int(255 * (1 - conf))
        g = int(255 * conf)
        b = 180
        return f"rgb({r},{g},{b})"

    lines, circles = [], []

    # Edges
    for a, b in COCO_EDGES:
        if a >= len(kpts_frame) or b >= len(kpts_frame):
            continue
        xa, ya, ca = kpts_frame[a]
        xb, yb, cb = kpts_frame[b]
        x1, y1 = c2s(xa, ya)
        x2, y2 = c2s(xb, yb)
        avg_conf = (ca + cb) / 2
        stroke = conf_color(avg_conf) if occlusion else "rgba(0,212,255,0.55)"
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="2.5" stroke-linecap="round"/>'
        )

    # Joints
    for idx, (x, y, conf) in enumerate(kpts_frame):
        cx, cy = c2s(x, y)
        color  = conf_color(conf)
        r      = 5 if idx in (0,) else 4   # nose slightly larger
        circles.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" '
            f'fill="{color}" stroke="#050d1a" stroke-width="1.5"/>'
        )

    svg = f"""
<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"
     style="background:#050d1a;border-radius:8px;width:100%;height:auto;">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="60%">
      <stop offset="0%"   stop-color="#0a1628"/>
      <stop offset="100%" stop-color="#050d1a"/>
    </radialGradient>
  </defs>
  <rect width="{width}" height="{height}" fill="url(#bg)"/>
  {''.join(lines)}
  {''.join(circles)}
</svg>"""
    return svg


# ─────────────────────────────────────────────────────────────
# PLOTLY PROBABILITY CHART
# ─────────────────────────────────────────────────────────────
def build_prob_chart(probs: list, threshold: float, label: str) -> go.Figure:
    frames  = list(range(len(probs)))
    color   = "#ff4b6e" if label == "Abnormal" else "#00ffc2"

    fig = go.Figure()

    # Threshold band
    fig.add_hrect(y0=threshold, y1=1.0, fillcolor="rgba(255,75,110,0.06)",
                  line_width=0, layer="below")

    # Smoothed area
    fig.add_trace(go.Scatter(
        x=frames, y=probs,
        fill="tozeroy",
        fillcolor=f"rgba({'255,75,110' if label=='Abnormal' else '0,255,194'},.10)",
        line=dict(color=color, width=2),
        mode="lines",
        name="Abnormality P",
        hovertemplate="Frame %{x}<br>P=%{y:.3f}<extra></extra>",
    ))

    # Threshold line
    fig.add_hline(y=threshold, line_dash="dot",
                  line_color="rgba(255,179,71,.7)", line_width=1.5,
                  annotation_text=f"τ={threshold}",
                  annotation_font_color="#ffb347",
                  annotation_position="top right")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#7fa8c9", size=11),
        margin=dict(l=0, r=10, t=10, b=0),
        height=240,
        xaxis=dict(
            title="Frame", showgrid=True,
            gridcolor="rgba(26,48,80,.6)", zeroline=False,
            title_font_color="#3a5a78", tickfont_color="#3a5a78",
        ),
        yaxis=dict(
            title="P(Abnormal)", range=[0, 1],
            showgrid=True, gridcolor="rgba(26,48,80,.6)", zeroline=False,
            title_font_color="#3a5a78", tickfont_color="#3a5a78",
        ),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# CONFIDENCE HEATMAP
# ─────────────────────────────────────────────────────────────
def build_confidence_heatmap(kpts: list) -> go.Figure:
    """Per-joint mean confidence across time."""
    arr = np.array(kpts)          # [T, 17, 3]
    mean_conf = arr[:, :, 2].mean(axis=0)   # [17]

    fig = go.Figure(go.Bar(
        x=COCO_KP_NAMES,
        y=mean_conf.tolist(),
        marker_color=[
            f"rgba(0,212,255,{0.3 + 0.7*c})" for c in mean_conf
        ],
        hovertemplate="%{x}<br>Conf: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#7fa8c9", size=10),
        margin=dict(l=0, r=0, t=10, b=0),
        height=180,
        xaxis=dict(showgrid=False, tickangle=-45,
                   tickfont=dict(size=9), zeroline=False),
        yaxis=dict(range=[0,1], showgrid=True,
                   gridcolor="rgba(26,48,80,.6)", zeroline=False),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.5rem 0 1.2rem'>
      <div style='font-family:Syne,sans-serif;font-size:1.35rem;font-weight:800;
                  color:#e8f4fd;letter-spacing:-.02em;'>GaitVision</div>
      <div style='font-family:DM Mono,monospace;font-size:.65rem;color:#3a5a78;
                  letter-spacing:.12em;text-transform:uppercase;margin-top:.15rem'>
        ICS555 · Ashesi University
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 📂 Input")
    uploaded = st.file_uploader(
        "Upload walking video",
        type=["mp4", "avi", "mov"],
        help="MP4 recommended. The video will be sent to Modal for processing.",
    )

    st.markdown("---")
    st.markdown("##### 🧠 Model")
    model_choice = st.radio(
        "Architecture",
        list(MODEL_CONFIGS.keys()),
        help="Switch models to compare performance.",
    )
    cfg = MODEL_CONFIGS[model_choice]
    st.markdown(f"""
    <div style='font-size:.72rem;color:#3a5a78;font-family:DM Mono,monospace;
                margin-top:-.4rem;margin-bottom:.8rem;line-height:1.5;'>
      {cfg['description']}<br>
      <span style='color:#1a3050;'>weights: {cfg['weight_path']}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### ⚙️ Parameters")
    threshold = st.slider(
        "Decision threshold τ",
        min_value=0.30, max_value=0.99,
        value=cfg["default_thresh"], step=0.01,
        help="Frames above τ are classified as Abnormal.",
    )

    st.markdown("---")
    st.markdown("##### 🔬 Analysis")
    occlusion_toggle = st.toggle(
        "Occlusion Sensitivity",
        value=False,
        help="Colour joints by detection confidence to visualise ablation findings.",
    )
    demo_mode = st.toggle(
        "Demo Mode (no Modal)",
        value=True,
        help="Use synthetic mock data instead of live Modal inference.",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.68rem;color:#3a5a78;font-family:DM Mono,monospace;
                line-height:1.7'>
      <div>Dataset: <a href='https://github.com/niais/mv-tgcn' 
           style='color:#1a3050;'>GAVD (1,874 seqs)</a></div>
      <div>Inference: Modal Serverless GPUs</div>
      <div>Pose: YOLOv8n-Pose (COCO-17)</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="gv-header">
  <h1>GaitVision</h1>
  <div class="sub">ST-GCN Automated Gait Analysis · Computer Vision Final Project</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────
if uploaded is None:
    col_a, col_b, col_c = st.columns([1,2,1])
    with col_b:
        st.markdown("""
        <div class="gv-card" style="text-align:center;padding:3rem 2rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">🦿</div>
          <div style="font-family:Syne,sans-serif;font-size:1.2rem;
                      font-weight:700;color:#e8f4fd;margin-bottom:.5rem;">
            Upload a walking video to begin
          </div>
          <div style="font-size:.82rem;color:#3a5a78;font-family:DM Mono,monospace;
                      line-height:1.8;">
            Supports MP4 · AVI · MOV<br>
            Pipeline: YOLOv8-Pose → ST-GCN<br>
            Classification: Normal / Abnormal
          </div>
        </div>

        <div class="gv-card" style="padding:1rem 1.5rem;">
          <div class="gv-card-title">Model Performance Summary (GAVD Test Set)</div>
          <div class="metric-row">
            <div class="metric-chip">
              <div class="val">0.899</div>
              <div class="lbl">AUC-ROC</div>
            </div>
            <div class="metric-chip">
              <div class="val">0.661</div>
              <div class="lbl">Macro F1</div>
            </div>
            <div class="metric-chip">
              <div class="val">1,874</div>
              <div class="lbl">Sequences</div>
            </div>
            <div class="metric-chip">
              <div class="val">85%</div>
              <div class="lbl">Threshold τ</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ─────────────────────────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────────────────────────
video_bytes = uploaded.read()

# Estimate frame count from file size (rough heuristic for display)
est_frames = max(30, min(120, len(video_bytes) // 50000))

run_btn = st.button("▶  Run Gait Analysis", use_container_width=False)

if "result" not in st.session_state:
    st.session_state.result = None
if "last_model" not in st.session_state:
    st.session_state.last_model = None

if run_btn or (st.session_state.result is not None and
               st.session_state.last_model == model_choice):
    if run_btn:
        with st.spinner("Modal GPU Spinning Up…  ☁️ This may take 20-40 s on cold start."):
            try:
                if demo_mode:
                    time.sleep(1.4)   # simulate latency
                    result = mock_inference(model_choice, n_frames=est_frames)
                else:
                    result = call_modal_inference(video_bytes, model_choice, threshold)
                st.session_state.result     = result
                st.session_state.last_model = model_choice
            except Exception as e:
                st.error(f"**Inference error:** {e}")
                st.stop()

result = st.session_state.result
if result is None:
    st.info("Click **Run Gait Analysis** to process the video.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# GUARD: low detection rate
# ─────────────────────────────────────────────────────────────
det_rate = result.get("detection_rate", 1.0)
if det_rate < 0.50:
    st.warning(
        f"⚠️  Detection rate is **{det_rate:.0%}** — below the 50% threshold "
        "required for reliable analysis (per GAVD data cleaning rules). "
        "Try a higher-quality video with unobstructed full-body footage."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# EXTRACT RESULT FIELDS
# ─────────────────────────────────────────────────────────────
label       = result["prediction_label"]
prob_score  = result["probability_score"]
probs       = result["prob_timeseries"]
kpts        = result["keypoint_tensor"]   # [T, 17, 3]
frame_conf  = result["frame_confidences"]
cadence     = result["cadence_steps_min"]
symmetry    = result["symmetry_index"]
n_frames    = result["n_frames"]

is_abnormal = label == "Abnormal"
badge_cls   = "badge-abnormal" if is_abnormal else "badge-normal"
score_pct   = prob_score * 100
bar_color   = "#ff4b6e" if is_abnormal else "#00ffc2"

# Pick mid-sequence frame for skeleton display
skel_frame_idx = min(len(kpts)//2, len(kpts)-1)
skel_data = kpts[skel_frame_idx]

# ─────────────────────────────────────────────────────────────
# DUAL-PANE DASHBOARD
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.35], gap="large")

# ── LEFT: Video + Skeleton ──────────────────────────────────
with left:
    st.markdown('<div class="gv-card-title">Input Video</div>', unsafe_allow_html=True)

    # Show original video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    st.video(tmp_path)
    os.unlink(tmp_path)

    # Skeleton overlay
    st.markdown('<div class="gv-card-title" style="margin-top:1rem;">Pose Skeleton · Mid-Sequence</div>',
                unsafe_allow_html=True)

    svg_html = render_skeleton_svg(skel_data, occlusion=occlusion_toggle)
    st.markdown(svg_html, unsafe_allow_html=True)

    if occlusion_toggle:
        st.markdown("""
        <div style='font-size:.7rem;color:#7fa8c9;font-family:DM Mono,monospace;
                    margin-top:.4rem;'>
          🎨 Occlusion mode: joints coloured by confidence channel.<br>
          <span style='color:#00ffc2'>Teal = high conf</span> · 
          <span style='color:#ff4b6e'>Red = low conf</span>
        </div>""", unsafe_allow_html=True)

# ── RIGHT: Analytics ────────────────────────────────────────
with right:

    # ── Classification result ──
    st.markdown(f"""
    <div class="gv-card">
      <div class="gv-card-title">Classification Result</div>
      <div style="display:flex;align-items:center;gap:1rem;margin-bottom:.75rem;">
        <div style="font-family:Syne,sans-serif;font-size:2.2rem;
                    font-weight:800;color:{'#ff4b6e' if is_abnormal else '#00ffc2'};">
          {label}
        </div>
        <span class="badge {badge_cls}">{score_pct:.1f}% confidence</span>
      </div>
      <div style="font-size:.78rem;color:#3a5a78;font-family:DM Mono,monospace;
                  margin-bottom:.6rem;">
        Model: {model_choice} · τ = {threshold:.2f} · Frames: {n_frames}
      </div>
      <div class="prob-bar-wrap">
        <div class="prob-bar-fill"
             style="width:{score_pct:.1f}%;background:{bar_color};"></div>
      </div>
      <div style="display:flex;justify-content:space-between;
                  font-family:DM Mono,monospace;font-size:.68rem;color:#1a3050;">
        <span>0%</span><span>τ={threshold:.0%}</span><span>100%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Gait metrics ──
    sym_color = "#00ffc2" if symmetry > 0.80 else "#ffb347" if symmetry > 0.65 else "#ff4b6e"
    cad_color = "#00ffc2" if 95 < cadence < 115 else "#ffb347"

    st.markdown(f"""
    <div class="gv-card">
      <div class="gv-card-title">Gait Feature Metrics</div>
      <div class="metric-row">
        <div class="metric-chip">
          <div class="val" style="color:{cad_color};">{cadence:.1f}</div>
          <div class="lbl">Cadence (steps/min)</div>
        </div>
        <div class="metric-chip">
          <div class="val" style="color:{sym_color};">{symmetry:.3f}</div>
          <div class="lbl">Symmetry Index</div>
        </div>
        <div class="metric-chip">
          <div class="val">{det_rate:.0%}</div>
          <div class="lbl">Detection Rate</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability timeseries ──
    st.markdown('<div class="gv-card-title">Abnormality Probability Over Time</div>',
                unsafe_allow_html=True)
    fig_prob = build_prob_chart(probs, threshold, label)
    st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})

    # ── Confidence heatmap (only in occlusion mode) ──
    if occlusion_toggle:
        st.markdown('<div class="gv-card-title" style="margin-top:.5rem;">Per-Joint Confidence (Ablation Channel 3)</div>',
                    unsafe_allow_html=True)
        fig_conf = build_confidence_heatmap(kpts)
        st.plotly_chart(fig_conf, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────
# MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="gv-card-title">Model Comparison · GAVD Test Set</div>', unsafe_allow_html=True)

comparison_data = {
    "Model":       ["ST-GCN (Main)", "1D-CNN Baseline", "Channel Ablation (XY only)"],
    "AUC-ROC":     ["0.899 ★", "0.761", "0.843"],
    "Macro F1":    ["0.661 ★", "0.512", "0.598"],
    "Accuracy":    ["73.4%", "61.2%", "68.9%"],
    "Params":      ["~0.8M", "~0.2M", "~0.8M"],
    "In-Channels": ["3 (x,y,conf)", "3 (x,y,conf)", "2 (x,y)"],
}

rows_html = ""
for i in range(3):
    is_current = comparison_data["Model"][i] == model_choice
    row_style = "background:rgba(0,212,255,.06);border-left:3px solid #00d4ff;" if is_current else ""
    rows_html += f"""
    <tr style="{row_style}">
      <td style="padding:.6rem 1rem;font-family:Syne,sans-serif;font-weight:{'700' if is_current else '400'};
                 color:{'#e8f4fd' if is_current else '#7fa8c9'};">
        {comparison_data['Model'][i]}
        {'<span class="badge badge-info" style="margin-left:.5rem;font-size:.62rem;">active</span>' if is_current else ''}
      </td>
      <td style="padding:.6rem 1rem;font-family:DM Mono,monospace;
                 color:{'#00d4ff' if is_current else '#7fa8c9'};">{comparison_data['AUC-ROC'][i]}</td>
      <td style="padding:.6rem 1rem;font-family:DM Mono,monospace;
                 color:{'#00d4ff' if is_current else '#7fa8c9'};">{comparison_data['Macro F1'][i]}</td>
      <td style="padding:.6rem 1rem;font-family:DM Mono,monospace;color:#3a5a78;">{comparison_data['Accuracy'][i]}</td>
      <td style="padding:.6rem 1rem;font-family:DM Mono,monospace;color:#3a5a78;">{comparison_data['Params'][i]}</td>
      <td style="padding:.6rem 1rem;font-family:DM Mono,monospace;color:#3a5a78;">{comparison_data['In-Channels'][i]}</td>
    </tr>"""

st.markdown(f"""
<div class="gv-card" style="overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;">
    <thead>
      <tr style="border-bottom:1px solid #1a3050;">
        <th style="padding:.5rem 1rem;text-align:left;font-family:DM Mono,monospace;
                   font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;
                   color:#3a5a78;font-weight:500;">Model</th>
        <th style="padding:.5rem 1rem;text-align:left;font-family:DM Mono,monospace;
                   font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;
                   color:#3a5a78;font-weight:500;">AUC-ROC</th>
        <th style="padding:.5rem 1rem;text-align:left;font-family:DM Mono,monospace;
                   font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;
                   color:#3a5a78;font-weight:500;">Macro F1</th>
        <th style="padding:.5rem 1rem;text-align:left;font-family:DM Mono,monospace;
                   font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;
                   color:#3a5a78;font-weight:500;">Accuracy</th>
        <th style="padding:.5rem 1rem;text-align:left;font-family:DM Mono,monospace;
                   font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;
                   color:#3a5a78;font-weight:500;">Params</th>
        <th style="padding:.5rem 1rem;text-align:left;font-family:DM Mono,monospace;
                   font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;
                   color:#3a5a78;font-weight:500;">Channels</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="gv-footer">
  <div class="stat-pill">AUC-ROC <span>0.899</span></div>
  <div class="stat-pill">Macro F1 <span>0.661</span></div>
  <div style="flex:1"></div>
  <div>
    Dataset: <a href="https://github.com/niais/mv-tgcn" target="_blank">GAVD (1,874 sequences)</a>
    &nbsp;·&nbsp;
    Inference: <a href="https://modal.com" target="_blank">Modal Serverless GPUs</a>
    &nbsp;·&nbsp;
    ICS555 Computer Vision · Ashesi University 2024
  </div>
</div>
""", unsafe_allow_html=True)