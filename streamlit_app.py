"""
GaitVision — ST-GCN Gait Analysis Dashboard
ICS555: Computer Vision | Ashesi University

Inference priority:
  1. Modal GPU  (if credentials present in st.secrets)
  2. Local CPU  (real models via inference.py — always available)
No synthetic mock data — the CPU path runs the actual models.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Make sure project modules are importable ─────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

:root {
  --bg-base:  #050d1a;
  --bg-card:  #0a1628;
  --bg-panel: #0d1e36;
  --accent:   #00d4ff;
  --accent2:  #00ffc2;
  --danger:   #ff4b6e;
  --warning:  #ffb347;
  --text-hi:  #e8f4fd;
  --text-mid: #7fa8c9;
  --text-lo:  #3a5a78;
  --border:   #1a3050;
  --radius:   8px;
  --glow:     0 0 20px rgba(0,212,255,.18);
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg-base) !important; color: var(--text-hi); }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

[data-testid="stSidebar"] {
  background: var(--bg-card) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-hi) !important; }
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSlider label { font-size:.82rem; color:var(--text-mid) !important; }

h1,h2,h3 { font-family:'Syne',sans-serif !important; letter-spacing:-.02em; }

.gv-card {
  background:var(--bg-card); border:1px solid var(--border);
  border-radius:var(--radius); padding:1.25rem 1.5rem;
  margin-bottom:1rem; box-shadow:var(--glow);
}
.gv-card-title {
  font-family:'DM Mono',monospace; font-size:.68rem;
  letter-spacing:.12em; text-transform:uppercase;
  color:var(--accent); margin-bottom:.6rem;
}

.metric-row { display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0; }
.metric-chip {
  flex:1; min-width:110px; background:var(--bg-panel);
  border:1px solid var(--border); border-radius:6px;
  padding:.8rem 1rem; text-align:center;
}
.metric-chip .val {
  font-family:'Syne',sans-serif; font-size:1.6rem;
  font-weight:700; color:var(--accent); line-height:1;
}
.metric-chip .lbl {
  font-size:.7rem; color:var(--text-mid); margin-top:.25rem;
  font-family:'DM Mono',monospace; letter-spacing:.06em;
}

.badge {
  display:inline-block; padding:.25rem .75rem; border-radius:4px;
  font-family:'DM Mono',monospace; font-size:.75rem;
  letter-spacing:.08em; font-weight:500;
}
.badge-normal   { background:rgba(0,255,194,.12); color:var(--accent2); border:1px solid rgba(0,255,194,.3); }
.badge-abnormal { background:rgba(255,75,110,.12); color:var(--danger);  border:1px solid rgba(255,75,110,.3); }
.badge-info     { background:rgba(0,212,255,.10);  color:var(--accent);  border:1px solid rgba(0,212,255,.25); }
.badge-cpu      { background:rgba(255,179,71,.10); color:var(--warning); border:1px solid rgba(255,179,71,.3); }
.badge-gpu      { background:rgba(0,255,194,.10);  color:var(--accent2); border:1px solid rgba(0,255,194,.3); }

.prob-bar-wrap { height:8px; background:var(--bg-panel); border-radius:4px; margin:.5rem 0; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:4px; transition:width .6s ease; }

.gv-header {
  display:flex; align-items:baseline; gap:1rem;
  border-bottom:1px solid var(--border);
  padding-bottom:1rem; margin-bottom:1.5rem;
}
.gv-header h1 { font-size:1.9rem; font-weight:800; color:var(--text-hi); margin:0; }
.gv-header .sub {
  font-family:'DM Mono',monospace; font-size:.72rem;
  color:var(--text-mid); letter-spacing:.1em; text-transform:uppercase;
}

.gv-footer {
  margin-top:3rem; padding-top:1.25rem;
  border-top:1px solid var(--border); font-size:.75rem;
  color:var(--text-lo); font-family:'DM Mono',monospace;
  display:flex; gap:2rem; flex-wrap:wrap; align-items:center;
}
.gv-footer a { color:var(--text-mid); text-decoration:none; }
.gv-footer a:hover { color:var(--accent); }
.stat-pill {
  display:inline-flex; gap:.4rem; align-items:center;
  background:var(--bg-card); border:1px solid var(--border);
  border-radius:4px; padding:.2rem .6rem;
}
.stat-pill span { color:var(--accent); font-weight:500; }

.stButton>button {
  background:linear-gradient(135deg,#00d4ff22,#00ffc211) !important;
  border:1px solid var(--accent) !important; color:var(--accent) !important;
  border-radius:6px !important; font-family:'DM Mono',monospace !important;
  letter-spacing:.06em !important; font-size:.82rem !important;
  transition:all .2s !important;
}
.stButton>button:hover {
  background:linear-gradient(135deg,#00d4ff44,#00ffc233) !important;
  box-shadow:var(--glow) !important;
}
.stRadio [data-baseweb="radio"] { accent-color:var(--accent); }
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background:var(--accent) !important; border-color:var(--accent) !important;
}
hr { border-color:var(--border) !important; }
[data-testid="stFileUploader"] {
  border:1px dashed var(--border) !important;
  border-radius:var(--radius) !important; background:var(--bg-panel) !important;
}
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg-base); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# COCO-17 SKELETON
# ─────────────────────────────────────────────────────────────
from src.skeleton import EDGES as COCO_EDGES, KEYPOINT_NAMES as COCO_KP_NAMES

# ─────────────────────────────────────────────────────────────
# MODEL CONFIGS
# ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "ST-GCN (Main)": {
        "weight_key":     "stgcn/runs/best_model.pt",
        "model_arch":     "stgcn",
        "in_channels":    3,
        "description":    "Spatio-Temporal GCN · COCO-17 · x,y,conf",
        "default_thresh": 0.85,
    },
    "1D-CNN Baseline": {
        "weight_key":     "baseline/runs/best_model.pt",
        "model_arch":     "baseline",
        "in_channels":    3,
        "description":    "1-D CNN baseline · flattened x,y timeseries",
        "default_thresh": 0.50,
    },
    "Channel Ablation": {
        "weight_key":     "stgcn/runs/ablation/best_model.pt",
        "model_arch":     "stgcn",
        "in_channels":    2,
        "description":    "ST-GCN ablation · XY only (conf channel excluded)",
        "default_thresh": 0.80,
    },
}

# ─────────────────────────────────────────────────────────────
# BACKEND HELPERS
# ─────────────────────────────────────────────────────────────
def _modal_available() -> bool:
    """Return True only if Modal secrets are present and the package is installed."""
    try:
        import modal  # noqa
        return bool(
            st.secrets.get("MODAL_TOKEN_ID") and
            st.secrets.get("MODAL_TOKEN_SECRET")
        )
    except Exception:
        return False


def call_modal(video_bytes: bytes, cfg: dict, threshold: float) -> dict:
    import modal
    os.environ["MODAL_TOKEN_ID"]     = st.secrets["MODAL_TOKEN_ID"]
    os.environ["MODAL_TOKEN_SECRET"] = st.secrets["MODAL_TOKEN_SECRET"]
    app = modal.App.lookup("gait-analysis")
    fn  = app.function("run_gait_inference")
    result = fn.remote(
        video_bytes  = video_bytes,
        weight_path  = cfg["weight_key"],
        model_folder = cfg["model_arch"],
        in_channels  = cfg["in_channels"],
        threshold    = threshold,
    )
    result["_backend"] = "modal_gpu"
    return result


def call_local_cpu(video_bytes: bytes, cfg: dict, threshold: float) -> dict:
    from inference import run_inference
    return run_inference(
        video_bytes = video_bytes,
        weight_key  = cfg["weight_key"],
        model_arch  = cfg["model_arch"],
        in_channels = cfg["in_channels"],
        threshold   = threshold,
    )


# ─────────────────────────────────────────────────────────────
# SKELETON SVG RENDERER
# ─────────────────────────────────────────────────────────────
def render_skeleton_svg(kpts_frame: list, occlusion: bool,
                        width: int = 400, height: int = 500) -> str:
    def c2s(x_n, y_n):
        # kpts are hip-centred normalised; remap to canvas
        x_c = (x_n * 0.35 + 0.5) * width
        y_c = (y_n * 0.25 + 0.5) * height
        return x_c, y_c

    def conf_color(conf: float) -> str:
        if not occlusion:
            return "#00d4ff"
        r = int(255 * (1 - conf))
        g = int(200 * conf)
        return f"rgb({r},{g},180)"

    lines, circles = [], []
    for a, b in COCO_EDGES:
        if a >= len(kpts_frame) or b >= len(kpts_frame):
            continue
        xa, ya, ca = kpts_frame[a]
        xb, yb, cb = kpts_frame[b]
        x1, y1 = c2s(xa, ya)
        x2, y2 = c2s(xb, yb)
        stroke = conf_color((ca + cb) / 2) if occlusion else "rgba(0,212,255,0.55)"
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="2.5" stroke-linecap="round"/>'
        )
    for idx, (x, y, conf) in enumerate(kpts_frame):
        cx, cy = c2s(x, y)
        color  = conf_color(conf)
        r = 5 if idx == 0 else 4
        circles.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" '
            f'fill="{color}" stroke="#050d1a" stroke-width="1.5"/>'
        )
    return f"""
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


# ─────────────────────────────────────────────────────────────
# PLOTLY CHARTS
# ─────────────────────────────────────────────────────────────
def build_prob_chart(probs: list, threshold: float, label: str) -> go.Figure:
    color = "#ff4b6e" if label == "Abnormal" else "#00ffc2"
    rgb   = "255,75,110" if label == "Abnormal" else "0,255,194"
    fig   = go.Figure()
    fig.add_hrect(y0=threshold, y1=1.0, fillcolor="rgba(255,75,110,0.06)",
                  line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=list(range(len(probs))), y=probs,
        fill="tozeroy", fillcolor=f"rgba({rgb},.10)",
        line=dict(color=color, width=2), mode="lines",
        hovertemplate="Frame %{x}<br>P=%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=threshold, line_dash="dot",
                  line_color="rgba(255,179,71,.7)", line_width=1.5,
                  annotation_text=f"τ={threshold}",
                  annotation_font_color="#ffb347",
                  annotation_position="top right")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#7fa8c9", size=11),
        margin=dict(l=0, r=10, t=10, b=0), height=240,
        xaxis=dict(title="Frame", showgrid=True, gridcolor="rgba(26,48,80,.6)",
                   zeroline=False, title_font_color="#3a5a78", tickfont_color="#3a5a78"),
        yaxis=dict(title="P(Abnormal)", range=[0,1], showgrid=True,
                   gridcolor="rgba(26,48,80,.6)", zeroline=False,
                   title_font_color="#3a5a78", tickfont_color="#3a5a78"),
        showlegend=False,
    )
    return fig


def build_confidence_heatmap(kpts: list) -> go.Figure:
    arr       = np.array(kpts)
    mean_conf = arr[:, :, 2].mean(axis=0)
    fig = go.Figure(go.Bar(
        x=COCO_KP_NAMES, y=mean_conf.tolist(),
        marker_color=[f"rgba(0,212,255,{0.3+0.7*c})" for c in mean_conf],
        hovertemplate="%{x}<br>Conf: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#7fa8c9", size=10),
        margin=dict(l=0, r=0, t=10, b=0), height=180,
        xaxis=dict(showgrid=False, tickangle=-45, tickfont=dict(size=9), zeroline=False),
        yaxis=dict(range=[0,1], showgrid=True, gridcolor="rgba(26,48,80,.6)", zeroline=False),
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
        help="MP4 recommended.",
    )

    st.markdown("---")
    st.markdown("##### 🧠 Model")
    model_choice = st.radio("Architecture", list(MODEL_CONFIGS.keys()))
    cfg = MODEL_CONFIGS[model_choice]
    st.markdown(f"""
    <div style='font-size:.72rem;color:#3a5a78;font-family:DM Mono,monospace;
                margin-top:-.4rem;margin-bottom:.8rem;line-height:1.5;'>
      {cfg['description']}<br>
      <span style='color:#1a3050;'>{cfg['weight_key']}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### ⚙️ Parameters")
    threshold = st.slider(
        "Decision threshold τ",
        min_value=0.30, max_value=0.99,
        value=cfg["default_thresh"], step=0.01,
    )

    st.markdown("---")
    st.markdown("##### 🔬 Visualisation")
    occlusion_toggle = st.toggle(
        "Occlusion Sensitivity",
        value=False,
        help="Colour joints by detection confidence.",
    )

    # ── Backend status indicator ──
    st.markdown("---")
    modal_ok = _modal_available()
    if modal_ok:
        st.markdown('<span class="badge badge-gpu">⚡ Modal GPU ready</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-cpu">🖥 CPU inference</span>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:.68rem;color:#3a5a78;font-family:DM Mono,monospace;
                    margin-top:.4rem;line-height:1.6'>
          Add MODAL_TOKEN_ID &amp;<br>MODAL_TOKEN_SECRET to<br>
          Streamlit Secrets for GPU.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.68rem;color:#3a5a78;font-family:DM Mono,monospace;line-height:1.7'>
      <div>Dataset: <a href='https://github.com/niais/mv-tgcn'
           style='color:#1a3050;'>GAVD (1,874 seqs)</a></div>
      <div>Pose: YOLOv8n-Pose (COCO-17)</div>
      <div>GPU: Modal Serverless</div>
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
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown("""
        <div class="gv-card" style="text-align:center;padding:3rem 2rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">🦿</div>
          <div style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;
                      color:#e8f4fd;margin-bottom:.5rem;">Upload a walking video to begin</div>
          <div style="font-size:.82rem;color:#3a5a78;font-family:DM Mono,monospace;line-height:1.8;">
            Supports MP4 · AVI · MOV<br>
            Pipeline: YOLOv8-Pose → ST-GCN<br>
            Classification: Normal / Abnormal
          </div>
        </div>
        <div class="gv-card" style="padding:1rem 1.5rem;">
          <div class="gv-card-title">Model Performance Summary (GAVD Test Set)</div>
          <div class="metric-row">
            <div class="metric-chip"><div class="val">0.899</div><div class="lbl">AUC-ROC</div></div>
            <div class="metric-chip"><div class="val">0.661</div><div class="lbl">Macro F1</div></div>
            <div class="metric-chip"><div class="val">1,874</div><div class="lbl">Sequences</div></div>
            <div class="metric-chip"><div class="val">85%</div><div class="lbl">Threshold τ</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────────────────────────
video_bytes = uploaded.read()

if st.button("▶  Run Gait Analysis"):
    backend_label = "Modal GPU" if modal_ok else "CPU"
    spinner_msg   = (
        "☁️  Sending to Modal GPU…  (cold start may take 20–40 s)"
        if modal_ok else
        "🖥  Running inference on CPU…  (may take a minute for long videos)"
    )

    with st.spinner(spinner_msg):
        modal_error = None
        backend_used = "cpu"

        if modal_ok:
            try:
                result = call_modal(video_bytes, cfg, threshold)
                backend_used = "modal_gpu"
            except Exception as e:
                modal_error  = str(e)
                # Fall through to CPU
                result = call_local_cpu(video_bytes, cfg, threshold)
                backend_used = "cpu_fallback"
        else:
            result = call_local_cpu(video_bytes, cfg, threshold)
            backend_used = "cpu"

        st.session_state.result       = result
        st.session_state.last_model   = model_choice
        st.session_state.modal_error  = modal_error
        st.session_state.backend_used = backend_used

# ─────────────────────────────────────────────────────────────
# GUARD: nothing run yet
# ─────────────────────────────────────────────────────────────
result = st.session_state.get("result")
if result is None:
    st.info("Click **▶ Run Gait Analysis** to process the uploaded video.")
    st.stop()

# ── Modal fell back to CPU ──
modal_error  = st.session_state.get("modal_error")
backend_used = st.session_state.get("backend_used", "cpu")
if modal_error:
    st.warning(
        f"⚠️  **Modal unreachable** — fell back to CPU inference automatically.  \n"
        f"*{modal_error}*",
        icon="☁️",
    )

# ── Low detection rate ──
det_rate = result.get("detection_rate", 1.0)
if result.get("error") == "low_detection" or det_rate < 0.50:
    st.warning(
        f"⚠️  Pose detection rate is **{det_rate:.0%}** — below the 50% minimum. "
        "Try a clearer video with an unobstructed full-body view."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# EXTRACT RESULT FIELDS
# ─────────────────────────────────────────────────────────────
label       = result["prediction_label"]
prob_score  = result["probability_score"]
probs       = result["prob_timeseries"]
kpts        = result["keypoint_tensor"]       # [T, 17, 3]
frame_conf  = result["frame_confidences"]
cadence     = result["cadence_steps_min"]
symmetry    = result["symmetry_index"]
n_frames    = result["n_frames"]

is_abnormal = label == "Abnormal"
badge_cls   = "badge-abnormal" if is_abnormal else "badge-normal"
score_pct   = prob_score * 100
bar_color   = "#ff4b6e" if is_abnormal else "#00ffc2"

skel_idx  = min(len(kpts) // 2, len(kpts) - 1)
skel_data = kpts[skel_idx]

backend_badge = (
    '<span class="badge badge-gpu">⚡ Modal GPU</span>'
    if backend_used == "modal_gpu" else
    '<span class="badge badge-cpu">🖥 CPU</span>'
)

# ─────────────────────────────────────────────────────────────
# DUAL-PANE DASHBOARD
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.35], gap="large")

# ── LEFT ─────────────────────────────────────────────────────
with left:
    st.markdown('<div class="gv-card-title">Input Video</div>', unsafe_allow_html=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    st.video(tmp_path)
    os.unlink(tmp_path)

    st.markdown(
        '<div class="gv-card-title" style="margin-top:1rem;">Pose Skeleton · Mid-Sequence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(render_skeleton_svg(skel_data, occlusion_toggle), unsafe_allow_html=True)

    if occlusion_toggle:
        st.markdown("""
        <div style='font-size:.7rem;color:#7fa8c9;font-family:DM Mono,monospace;margin-top:.4rem;'>
          🎨 Joints coloured by confidence channel ·
          <span style='color:#00ffc2'>teal = high</span> ·
          <span style='color:#ff4b6e'>red = low</span>
        </div>""", unsafe_allow_html=True)

# ── RIGHT ─────────────────────────────────────────────────────
with right:
    # Classification result
    st.markdown(f"""
    <div class="gv-card">
      <div class="gv-card-title">Classification Result</div>
      <div style="display:flex;align-items:center;gap:1rem;margin-bottom:.75rem;">
        <div style="font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;
                    color:{'#ff4b6e' if is_abnormal else '#00ffc2'};">{label}</div>
        <span class="badge {badge_cls}">{score_pct:.1f}% confidence</span>
        {backend_badge}
      </div>
      <div style="font-size:.78rem;color:#3a5a78;font-family:DM Mono,monospace;
                  margin-bottom:.6rem;">
        Model: {model_choice} · τ = {threshold:.2f} · Frames: {n_frames}
      </div>
      <div class="prob-bar-wrap">
        <div class="prob-bar-fill" style="width:{score_pct:.1f}%;background:{bar_color};"></div>
      </div>
      <div style="display:flex;justify-content:space-between;
                  font-family:DM Mono,monospace;font-size:.68rem;color:#1a3050;">
        <span>0%</span><span>τ={threshold:.0%}</span><span>100%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Gait metrics
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

    # Probability timeseries
    st.markdown('<div class="gv-card-title">Abnormality Probability Over Time</div>',
                unsafe_allow_html=True)
    st.plotly_chart(build_prob_chart(probs, threshold, label),
                    use_container_width=True, config={"displayModeBar": False})

    # Confidence heatmap (occlusion mode only)
    if occlusion_toggle and kpts:
        st.markdown(
            '<div class="gv-card-title" style="margin-top:.5rem;">'
            'Per-Joint Confidence (Ablation Channel 3)</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_confidence_heatmap(kpts),
                        use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────
# MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="gv-card-title">Model Comparison · GAVD Test Set</div>',
            unsafe_allow_html=True)

rows_data = [
    ("ST-GCN (Main)",        "0.899 ★", "0.661 ★", "73.4%", "~0.8M", "3 (x,y,conf)"),
    ("1D-CNN Baseline",      "0.761",   "0.512",   "61.2%", "~0.2M", "3 (x,y,conf)"),
    ("Channel Ablation",     "0.843",   "0.598",   "68.9%", "~0.8M", "2 (x,y)"),
]
headers = ["Model", "AUC-ROC", "Macro F1", "Accuracy", "Params", "Channels"]

rows_html = ""
for row in rows_data:
    active = row[0] == model_choice
    rs = "background:rgba(0,212,255,.06);border-left:3px solid #00d4ff;" if active else ""
    fw = "700" if active else "400"
    tc = "#e8f4fd" if active else "#7fa8c9"
    vc = "#00d4ff" if active else "#7fa8c9"
    rows_html += f"""
    <tr style="{rs}">
      <td style="padding:.6rem 1rem;font-family:Syne,sans-serif;font-weight:{fw};color:{tc};">
        {row[0]}
        {'<span class="badge badge-info" style="margin-left:.5rem;font-size:.62rem;">active</span>' if active else ''}
      </td>
      <td style="padding:.6rem 1rem;font-family:DM Mono,monospace;color:{vc};">{row[1]}</td>
      <td style="padding:.6rem 1rem;font-family:DM Mono,monospace;color:{vc};">{row[2]}</td>
      <td style="padding:.6rem 1rem;font-family:DM Mono,monospace;color:#3a5a78;">{row[3]}</td>
      <td style="padding:.6rem 1rem;font-family:DM Mono,monospace;color:#3a5a78;">{row[4]}</td>
      <td style="padding:.6rem 1rem;font-family:DM Mono,monospace;color:#3a5a78;">{row[5]}</td>
    </tr>"""

th_style = ("padding:.5rem 1rem;text-align:left;font-family:DM Mono,monospace;"
            "font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;"
            "color:#3a5a78;font-weight:500;")
header_row = "".join(f'<th style="{th_style}">{h}</th>' for h in headers)

st.markdown(f"""
<div class="gv-card" style="overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;">
    <thead><tr style="border-bottom:1px solid #1a3050;">{header_row}</tr></thead>
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
    GPU: <a href="https://modal.com" target="_blank">Modal Serverless</a>
    &nbsp;·&nbsp;
    ICS555 Computer Vision · Ashesi University 2024
  </div>
</div>
""", unsafe_allow_html=True)