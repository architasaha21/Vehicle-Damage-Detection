# ============================================================
# app.py — Vehicle Damage Detection Streamlit Dashboard
# Run: streamlit run app.py
# ============================================================

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.detector import load_model, run_detection, generate_gradcam, severity_score

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="VisionDamage AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark industrial theme ────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111318 0%, #0d0f14 100%);
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Header */
.app-header {
    background: linear-gradient(135deg, #0d0f14 0%, #111827 100%);
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.app-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.app-subtitle {
    color: #64748b;
    font-size: 0.85rem;
    margin: 0;
    font-family: 'Space Mono', monospace;
}

/* Cards */
.card {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.75rem;
}

/* Damage badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
    margin: 2px;
}
.badge-scratch      { background: #0e3a4a; color: #38bdf8; border: 1px solid #38bdf8; }
.badge-dent         { background: #3a2000; color: #f97316; border: 1px solid #f97316; }
.badge-broken_part  { background: #3a0000; color: #f87171; border: 1px solid #f87171; }
.badge-paint_damage { background: #2d1a3a; color: #c084fc; border: 1px solid #c084fc; }

/* Severity pill */
.sev-low      { background: #14532d; color: #4ade80; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.sev-medium   { background: #3a2000; color: #fbbf24; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.sev-high     { background: #3a0000; color: #f87171; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

/* Risk flag */
.risk-flag {
    background: linear-gradient(135deg, #3a0000, #1a0000);
    border: 1px solid #f87171;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #f87171;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}
.safe-flag {
    background: linear-gradient(135deg, #052e16, #021a0c);
    border: 1px solid #4ade80;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #4ade80;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}

/* Detection row */
.detection-row {
    background: #0d1117;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Confidence bar */
.conf-bar-wrap { background: #1e2330; border-radius: 4px; height: 6px; width: 100px; display: inline-block; vertical-align: middle; }
.conf-bar-fill { border-radius: 4px; height: 6px; background: linear-gradient(90deg, #38bdf8, #818cf8); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111318;
    border-radius: 8px;
    gap: 4px;
    padding: 4px;
    border: 1px solid #1e2330;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}
.stTabs [aria-selected="true"] {
    background: #1e2330 !important;
    color: #38bdf8 !important;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #111318;
    border: 2px dashed #1e2330;
    border-radius: 12px;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8;
}

/* Sliders */
.stSlider [data-baseweb="slider"] { color: #38bdf8; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    padding: 0.6rem 1.5rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Metric cards */
.metric-card {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* JSON viewer */
.json-box {
    background: #0a0c10;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #a5f3fc;
    white-space: pre-wrap;
    overflow-x: auto;
    max-height: 400px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.1rem;
                    background: linear-gradient(90deg,#38bdf8,#818cf8);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    font-weight: 700;'>⬡ VisionDamage AI</div>
        <div style='color: #334155; font-size: 0.72rem; margin-top: 4px;'>v1.0 — Hero VIDA Challenge</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Model")

    model_path = st.text_input(
        "Model path (.pt)",
        value="model/best.pt",
        help="Path to your trained YOLOv8 best.pt file",
    )

    st.markdown("### 🎛️ Detection Settings")
    conf_threshold = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
    iou_threshold  = st.slider("IoU threshold (NMS)", 0.10, 0.90, 0.60, 0.05)

    st.markdown("### 🖼️ Display")
    show_gradcam = st.toggle("Show Grad-CAM heatmap", value=True)
    show_json    = st.toggle("Show raw JSON output",  value=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:#334155; font-size:0.7rem; font-family: Space Mono, monospace;'>
    Classes detected:<br>
    🔵 scratch &nbsp; 🟠 dent<br>
    🔴 broken_part &nbsp; 🟣 paint_damage
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <p class="app-title">🔍 VisionDamage AI</p>
        <p class="app-subtitle">AI-Driven Vehicle Damage Detection & Intelligent Assessment System</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(path):
    return load_model(path)

model = None
model_loaded = False

try:
    with st.spinner("Loading model..."):
        model = get_model(model_path)
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load model from `{model_path}`. Make sure `best.pt` is in the correct folder.\n\n`{e}`")


# ── Upload ────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">📤 Upload Vehicle Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drag & drop or click to upload",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)


# ── Main logic ────────────────────────────────────────────────
if uploaded and model_loaded:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("🔍 Analysing damage..."):
        annotated_img, report  = run_detection(model, image, conf_threshold, iou_threshold)
        gradcam_img            = generate_gradcam(model, image) if show_gradcam else None

    damages  = report["damages"]
    n_damage = report["total_damages_detected"]
    risk     = report["high_risk_flag"]
    sev_val  = severity_score(damages)

    # ── Top metrics row ───────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_damage}</div>
            <div class="metric-label">Damages Found</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        sev_color = "#f87171" if sev_val > 65 else "#fbbf24" if sev_val > 35 else "#4ade80"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{sev_color}">{sev_val:.0f}</div>
            <div class="metric-label">Severity Score /100</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        functional = sum(1 for d in damages if d["impact"] == "Functional")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#f97316">{functional}</div>
            <div class="metric-label">Functional Damages</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        avg_conf = np.mean([d["confidence_score"] for d in damages]) if damages else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_conf:.0%}</div>
            <div class="metric-label">Avg Confidence</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk flag ─────────────────────────────────────────────
    if risk:
        st.markdown('<div class="risk-flag">⚠️ HIGH RISK — Manual review recommended. High severity damage or low-confidence detections found.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="safe-flag">✅ LOW RISK — No critical damage flags triggered.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Image tabs ────────────────────────────────────────────
    tab_labels = ["📸 Original", "🎯 Detections"]
    if show_gradcam:
        tab_labels.append("🔥 Grad-CAM")

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        st.image(image, use_container_width=True, caption="Original uploaded image")

    with tabs[1]:
        st.image(annotated_img, use_container_width=True,
                 caption=f"{n_damage} damage(s) detected")

    if show_gradcam and gradcam_img and len(tabs) > 2:
        with tabs[2]:
            st.image(gradcam_img, use_container_width=True,
                     caption="Grad-CAM activation heatmap — red = high model attention")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two column layout: gauge + detections ─────────────────
    left, right = st.columns([1, 1.6])

    with left:
        st.markdown('<div class="card"><div class="card-title">📊 Severity Gauge</div>', unsafe_allow_html=True)

        # Plotly gauge
        gauge_color = "#f87171" if sev_val > 65 else "#fbbf24" if sev_val > 35 else "#4ade80"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sev_val,
            number={"font": {"color": gauge_color, "size": 36, "family": "Space Mono"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#334155",
                    "tickfont": {"color": "#64748b", "size": 10},
                },
                "bar":  {"color": gauge_color, "thickness": 0.25},
                "bgcolor": "#0d0f14",
                "bordercolor": "#1e2330",
                "steps": [
                    {"range": [0,  35],  "color": "#052e16"},
                    {"range": [35, 65],  "color": "#3a2000"},
                    {"range": [65, 100], "color": "#3a0000"},
                ],
                "threshold": {
                    "line": {"color": gauge_color, "width": 3},
                    "thickness": 0.75,
                    "value": sev_val,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="#111318",
            plot_bgcolor="#111318",
            margin=dict(t=20, b=10, l=20, r=20),
            height=240,
            font={"color": "#e2e8f0"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Severity legend
        st.markdown("""
        <div style='font-size:0.72rem; color:#64748b; font-family: Space Mono, monospace;
                    display:flex; gap:12px; justify-content:center; margin-top:-8px;'>
            <span style='color:#4ade80'>■ 0–35 Low</span>
            <span style='color:#fbbf24'>■ 35–65 Med</span>
            <span style='color:#f87171'>■ 65–100 High</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><div class="card-title">🔎 Detection Breakdown</div>', unsafe_allow_html=True)

        if not damages:
            st.markdown("<div style='color:#64748b; text-align:center; padding:2rem;'>No damage detected above confidence threshold.</div>", unsafe_allow_html=True)
        else:
            for i, d in enumerate(damages):
                sev_class = f"sev-{d['severity'].lower()}"
                badge_cls = f"badge-{d['damage_type']}"
                bar_width = int(d['confidence_score'] * 100)

                st.markdown(f"""
                <div class="detection-row">
                    <div style='display:flex; align-items:center; gap:10px;'>
                        <span style='color:#64748b; font-family:Space Mono,monospace; font-size:0.7rem;'>#{i+1}</span>
                        <span class="badge {badge_cls}">{d['damage_type']}</span>
                        <span class="{sev_class}">{d['severity']}</span>
                        <span style='font-size:0.72rem; color:{"#f87171" if d["impact"]=="Functional" else "#94a3b8"}'>
                            {"⚡ Functional" if d["impact"]=="Functional" else "✦ Cosmetic"}
                        </span>
                    </div>
                    <div style='text-align:right;'>
                        <div style='font-family:Space Mono,monospace; font-size:0.8rem; color:#e2e8f0;'>
                            {d['confidence_score']:.0%}
                        </div>
                        <div class="conf-bar-wrap">
                            <div class="conf-bar-fill" style="width:{bar_width}px; max-width:100px;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Class distribution bar chart ──────────────────────────
    if damages:
        st.markdown('<div class="card"><div class="card-title">📈 Class Distribution</div>', unsafe_allow_html=True)

        from collections import Counter
        cls_counts = Counter(d["damage_type"] for d in damages)
        colors_map = {
            "scratch":      "#38bdf8",
            "dent":         "#f97316",
            "broken_part":  "#f87171",
            "paint_damage": "#c084fc",
        }

        fig2 = go.Figure(go.Bar(
            x=list(cls_counts.keys()),
            y=list(cls_counts.values()),
            marker_color=[colors_map.get(k, "#64748b") for k in cls_counts.keys()],
            marker_line_width=0,
        ))
        fig2.update_layout(
            paper_bgcolor="#111318",
            plot_bgcolor="#111318",
            margin=dict(t=10, b=10, l=10, r=10),
            height=180,
            font={"color": "#64748b", "family": "Space Mono", "size": 11},
            xaxis={"gridcolor": "#1e2330", "tickfont": {"color": "#94a3b8"}},
            yaxis={"gridcolor": "#1e2330", "tickfont": {"color": "#94a3b8"}},
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── JSON output ───────────────────────────────────────────
    if show_json:
        st.markdown('<div class="card"><div class="card-title">{ } Raw JSON Output</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="json-box">{json.dumps(report, indent=2)}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            label="⬇️ Download JSON Report",
            data=json.dumps(report, indent=2),
            file_name="damage_report.json",
            mime="application/json",
        )
        st.markdown('</div>', unsafe_allow_html=True)

elif not model_loaded:
    # Show placeholder when model not loaded
    st.markdown("""
    <div style='text-align:center; padding:4rem 2rem; color:#334155;'>
        <div style='font-size:3rem; margin-bottom:1rem;'>⬡</div>
        <div style='font-family: Space Mono, monospace; font-size:1rem; color:#475569;'>
            Load a model to begin
        </div>
        <div style='font-size:0.8rem; margin-top:0.5rem;'>
            Set the model path in the sidebar → place your <code>best.pt</code> in the <code>model/</code> folder
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Show placeholder when no image uploaded
    st.markdown("""
    <div style='text-align:center; padding:4rem 2rem;'>
        <div style='font-size:3rem; margin-bottom:1rem;'>🚗</div>
        <div style='font-family: Space Mono, monospace; font-size:1rem; color:#475569;'>
            Upload a vehicle image to begin analysis
        </div>
        <div style='font-size:0.8rem; color:#334155; margin-top:0.5rem;'>
            Supports JPG, PNG, WEBP — any angle, any lighting
        </div>
    </div>
    """, unsafe_allow_html=True)
