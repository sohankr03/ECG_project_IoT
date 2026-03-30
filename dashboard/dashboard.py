"""
dashboard.py
─────────────────────────────────────────────────────────────────────────────
Streamlit Dashboard — AI-Based Real-Time ECG Anomaly Detection System

Run:
  cd E:\Project\ECG_Project
  streamlit run dashboard/dashboard.py

Features:
  - Live ECG waveform (last 5s, display at ~83 Hz, refresh every 500ms)
  - BPM gauge (color-coded: green/yellow/red)
  - HRV metric cards (SDNN, RMSSD, RR Mean)
  - AI prediction banner (Normal / ABNORMAL)
  - Feature values table
  - Alert history panel with timestamps
  - Electrode status indicator
  - Signal Quality Index (SQI) bar
  - Demo Mode (no hardware needed) via sidebar toggle
  - Calibration button (30s baseline capture)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import threading
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Path Setup ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

# ── Page Config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="ECG Anomaly Detection",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# Theme / Global CSS
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0E1A;
    color: #E2E8F0;
}

/* ── Main container ── */
.block-container {
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 100%;
}

/* ── Header ── */
.ecg-header {
    background: linear-gradient(135deg, #0D1B3E 0%, #12263F 50%, #0D1B3E 100%);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 4px 24px rgba(0, 212, 255, 0.08);
}
.ecg-header h1 {
    font-size: 1.7rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00D4FF, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.ecg-header .subtitle {
    font-size: 0.85rem;
    color: #64748B;
    margin-top: 0.2rem;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1A2332 100%);
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    height: 100%;
}
.metric-card:hover {
    border-color: #00D4FF44;
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.12);
}
.metric-card .metric-label {
    font-size: 0.75rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
    font-weight: 500;
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-card .metric-unit {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-top: 0.3rem;
}

/* ── Prediction banner ── */
.prediction-normal {
    background: linear-gradient(135deg, #052E16 0%, #064E3B 100%);
    border: 2px solid #10B981;
    border-radius: 14px;
    padding: 1.2rem 1.8rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(16, 185, 129, 0.20);
}
.prediction-abnormal {
    background: linear-gradient(135deg, #1C0A0A 0%, #2D1515 100%);
    border: 2px solid #EF4444;
    border-radius: 14px;
    padding: 1.2rem 1.8rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(239, 68, 68, 0.30);
    animation: pulse-red 1.5s ease-in-out infinite;
}
.prediction-waiting {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border: 2px solid #334155;
    border-radius: 14px;
    padding: 1.2rem 1.8rem;
    text-align: center;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 4px 24px rgba(239, 68, 68, 0.30); }
    50%       { box-shadow: 0 4px 40px rgba(239, 68, 68, 0.60); }
}
.prediction-label {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.prediction-prob {
    font-size: 0.9rem;
    color: #94A3B8;
}

/* ── Status badge ── */
.status-ok         { color: #10B981; font-weight: 600; }
.status-warning    { color: #F59E0B; font-weight: 600; }
.status-error      { color: #EF4444; font-weight: 600; }
.status-calibrating{ color: #A78BFA; font-weight: 600; }

/* ── Section headers ── */
.section-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 0.4rem;
}

/* ── Alert item ── */
.alert-item {
    background: #1C0A0A;
    border-left: 3px solid #EF4444;
    border-radius: 0 8px 8px 0;
    padding: 0.5rem 0.9rem;
    margin-bottom: 0.4rem;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    color: #FCA5A5;
}
.alert-item-normal {
    background: #052E16;
    border-left: 3px solid #10B981;
    border-radius: 0 8px 8px 0;
    padding: 0.5rem 0.9rem;
    margin-bottom: 0.4rem;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    color: #6EE7B7;
}

/* ── Feature table ── */
.feature-row {
    display: flex;
    justify-content: space-between;
    padding: 0.45rem 0;
    border-bottom: 1px solid #1E293B;
    font-size: 0.88rem;
}
.feature-row:last-child { border-bottom: none; }
.feature-name { color: #94A3B8; }
.feature-val  { font-family: 'JetBrains Mono', monospace; color: #E2E8F0; font-weight: 500; }

/* ── SQI bar ── */
.sqi-bar-wrap {
    background: #1E293B;
    border-radius: 100px;
    height: 8px;
    margin-top: 0.3rem;
    overflow: hidden;
}
.sqi-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.5s ease;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #070C17;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] .sidebar-title {
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin: 1.5rem 0 0.5rem 0;
}

/* ── Plotly chart frame ── */
.ecg-chart-wrap {
    background: #0D1117;
    border: 1px solid #1E293B;
    border-radius: 14px;
    overflow: hidden;
    padding: 0.5rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# Session State Initialisation
# ══════════════════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "engine"          : None,
        "engine_running"  : False,
        "demo_mode"       : True,
        "com_port"        : "COM3",
        "alert_history"   : [],   # list of dicts {time, label, prob}
        "refresh_interval": 500,  # ms between Streamlit reruns
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ══════════════════════════════════════════════════════════════════════════
# Engine Management
# ══════════════════════════════════════════════════════════════════════════

def start_engine():
    """Start the ECG inference engine (demo or real)."""
    from realtime_inference import ECGInferenceEngine

    if st.session_state["demo_mode"]:
        from ecg_simulator import EcgSimulator
        sim = EcgSimulator(record_name="119", inject_anomaly=True, loop=True)
        engine = ECGInferenceEngine(demo_mode=True, demo_stream=sim)
    else:
        port = st.session_state["com_port"]
        engine = ECGInferenceEngine(port=port, demo_mode=False)

    engine.start()
    st.session_state["engine"]         = engine
    st.session_state["engine_running"] = True


def stop_engine():
    if st.session_state["engine"] is not None:
        st.session_state["engine"].stop()
        st.session_state["engine"]         = None
        st.session_state["engine_running"] = False


# ══════════════════════════════════════════════════════════════════════════
# BPM Gauge
# ══════════════════════════════════════════════════════════════════════════

def bpm_color(bpm: float) -> str:
    if bpm <= 0:
        return "#475569"
    if bpm < 60:
        return "#38BDF8"   # bradycardia — blue
    if bpm <= 100:
        return "#10B981"   # normal — green
    if bpm <= 120:
        return "#F59E0B"   # mild tachycardia — yellow
    return "#EF4444"       # high — red


def make_bpm_gauge(bpm: float) -> go.Figure:
    color = bpm_color(bpm)
    display_bpm = max(0, min(bpm, 220))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_bpm,
        number={"suffix": " BPM", "font": {"size": 32, "color": color, "family": "JetBrains Mono"}},
        gauge={
            "axis"      : {"range": [0, 200], "tickcolor": "#475569", "tickwidth": 2,
                           "tickfont": {"color": "#64748B", "size": 11}},
            "bar"       : {"color": color, "thickness": 0.22},
            "bgcolor"   : "#0D1117",
            "borderwidth": 0,
            "steps"     : [
                {"range": [0, 60],   "color": "#0F1827"},
                {"range": [60, 100], "color": "#0A1F18"},
                {"range": [100, 120],"color": "#1A1505"},
                {"range": [120, 200],"color": "#1A0505"},
            ],
            "threshold" : {"line": {"color": color, "width": 3}, "value": display_bpm},
        },
    ))
    fig.update_layout(
        height=200, margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor="#0D1117", font_color="#E2E8F0",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
# ECG Waveform Chart
# ══════════════════════════════════════════════════════════════════════════

def make_ecg_chart(ecg_buffer: list, status: str) -> go.Figure:
    n = len(ecg_buffer)
    if n == 0:
        ecg_buffer = [2048.0]
        n = 1

    # X-axis: time in seconds (last 5s)
    display_fs = 250 / 3   # ~83 Hz display rate
    duration   = n / display_fs
    t = np.linspace(max(0, duration - 5), duration, n)

    # Normalise for display
    arr = np.array(ecg_buffer, dtype=float)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr_norm = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr_norm = np.zeros_like(arr)

    line_color = "#00D4FF" if status == "OK" else "#EF4444"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=arr_norm,
        mode="lines",
        line=dict(color=line_color, width=1.8, shape="spline", smoothing=0.3),
        fill="tozeroy",
        fillcolor=f"rgba(0, 212, 255, 0.04)" if status == "OK" else "rgba(239, 68, 68, 0.04)",
        name="ECG",
    ))

    # Grid lines
    for y_val in [0.25, 0.5, 0.75]:
        fig.add_hline(y=y_val, line_dash="dot", line_color="#1E293B", line_width=1)

    fig.update_layout(
        height=220,
        margin=dict(t=10, b=30, l=40, r=10),
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        showlegend=False,
        xaxis=dict(
            title="Time (s)",
            color="#475569",
            gridcolor="#1E293B",
            zeroline=False,
            range=[max(0, duration - 5), duration + 0.1],
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Amplitude (norm.)",
            color="#475569",
            gridcolor="#1E293B",
            zeroline=False,
            range=[-0.05, 1.1],
            tickfont=dict(size=10),
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 💓 ECG System")
        st.markdown("---")

        st.markdown('<p class="sidebar-title">Mode</p>', unsafe_allow_html=True)
        demo = st.toggle("Demo Mode (no hardware)", value=st.session_state["demo_mode"], key="demo_toggle")
        st.session_state["demo_mode"] = demo

        if not demo:
            st.markdown('<p class="sidebar-title">Hardware</p>', unsafe_allow_html=True)
            port = st.text_input("Serial Port", value=st.session_state["com_port"], key="port_input")
            st.session_state["com_port"] = port
            st.caption("e.g. COM3 (Windows) or /dev/ttyUSB0 (Linux)")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state["engine_running"]:
                if st.button("▶ Start", use_container_width=True, type="primary"):
                    try:
                        start_engine()
                        st.success("Engine started!")
                    except FileNotFoundError:
                        st.error("Model not found. Run model training first.")
                    except Exception as e:
                        st.error(f"Error: {e}")
        with col2:
            if st.session_state["engine_running"]:
                if st.button("⏹ Stop", use_container_width=True):
                    stop_engine()
                    st.info("Engine stopped.")

        st.markdown("---")
        st.markdown('<p class="sidebar-title">Calibration</p>', unsafe_allow_html=True)
        if st.button("🎯 Calibrate (30s)", use_container_width=True,
                     disabled=not st.session_state["engine_running"]):
            engine = st.session_state["engine"]
            if engine:
                engine.start_calibration()
                st.success("Calibration started…")

        st.markdown("---")
        st.markdown('<p class="sidebar-title">Display</p>', unsafe_allow_html=True)
        refresh = st.slider("Refresh interval (ms)", 200, 2000, 500, step=100)
        st.session_state["refresh_interval"] = refresh

        st.markdown("---")
        st.markdown("**About**")
        st.caption(
            "AI-Based Real-Time ECG Anomaly Detection  \n"
            "ESP32 + AD8232 · Random Forest  \n"
            "Final Year Project"
        )


# ══════════════════════════════════════════════════════════════════════════
# Main Dashboard Layout
# ══════════════════════════════════════════════════════════════════════════

def render_dashboard():
    engine = st.session_state["engine"]
    running = st.session_state["engine_running"]

    # ── Fetch current state ───────────────────────────────────────────
    if engine and running:
        ecg_buf    = engine.get_ecg_buffer()
        pred       = engine.get_latest_prediction()
        features   = engine.get_latest_features()
        status     = engine.get_status()
    else:
        ecg_buf  = []
        pred     = {"label": "—", "probability": 0.0, "consecutive_count": 0, "timestamp": "—"}
        features = {}
        status   = "STARTING" if not running else "NO_DATA"

    # ── Update alert history ──────────────────────────────────────────
    if running and pred["label"] not in ("—", "Poor Signal"):
        history = st.session_state["alert_history"]
        ts = pred["timestamp"]
        # Only add if different from last entry
        if not history or history[-1]["time"] != ts:
            history.append({
                "time"  : ts,
                "label" : pred["label"],
                "prob"  : pred["probability"],
            })
            # Keep last 20 entries
            st.session_state["alert_history"] = history[-20:]

    # ── Page Header ───────────────────────────────────────────────────
    mode_badge = (
        "🟡 <b>DEMO MODE</b>" if st.session_state["demo_mode"]
        else "🟢 <b>LIVE — ESP32</b>"
    )
    st.markdown(f"""
    <div class="ecg-header">
        <div style="font-size:2.5rem">💓</div>
        <div>
            <h1>ECG Anomaly Detection System</h1>
            <div class="subtitle">{mode_badge} &nbsp;·&nbsp; Random Forest Classifier &nbsp;·&nbsp; 250 Hz Sampling</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Status Bar ────────────────────────────────────────────────────
    status_map = {
        "OK"                    : ("🟢", "status-ok",          "Signal OK"),
        "ELECTRODE_DISCONNECTED": ("🔴", "status-error",       "Electrode Disconnected — Inference Paused"),
        "POOR_SIGNAL"           : ("🟡", "status-warning",     "Poor Signal Quality — Inference Skipped"),
        "CALIBRATING"           : ("🔵", "status-calibrating", "Calibrating (30s baseline)…"),
        "STARTING"              : ("⚪", "status-warning",     "Engine not started"),
        "NO_DATA"               : ("🔴", "status-error",       "No data — check serial connection"),
    }
    icon, cls, msg = status_map.get(status, ("⚪", "status-warning", status))
    sqi_val = features.get("sqi", 0.0) if features else 0.0
    st.markdown(
        f'<span class="{cls}">{icon} {msg}</span>'
        f'&nbsp;&nbsp;&nbsp;<span style="color:#475569;font-size:0.85rem">SQI: {sqi_val:.0%}</span>',
        unsafe_allow_html=True
    )
    st.markdown("")

    # ══════════════════════════════════════════════════════════════════
    # Row 1: ECG Chart (wide) + Prediction Banner
    # ══════════════════════════════════════════════════════════════════
    col_ecg, col_pred = st.columns([3, 1], gap="medium")

    with col_ecg:
        st.markdown('<div class="section-title">Live ECG Waveform</div>', unsafe_allow_html=True)
        st.markdown('<div class="ecg-chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(
            make_ecg_chart(ecg_buf, status),
            use_container_width=True, config={"displayModeBar": False}
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_pred:
        st.markdown('<div class="section-title">AI Prediction</div>', unsafe_allow_html=True)
        label = pred["label"]
        prob  = pred["probability"]
        cons  = pred["consecutive_count"]

        if label == "ABNORMAL":
            banner_cls = "prediction-abnormal"
            icon_pred  = "🚨"
            label_text = "ABNORMAL"
            label_color = "#EF4444"
        elif label == "Normal":
            banner_cls = "prediction-normal"
            icon_pred  = "✅"
            label_text = "Normal"
            label_color = "#10B981"
        else:
            banner_cls = "prediction-waiting"
            icon_pred  = "⏳"
            label_text = label
            label_color = "#64748B"

        st.markdown(f"""
        <div class="{banner_cls}">
            <div style="font-size:2.2rem">{icon_pred}</div>
            <div class="prediction-label" style="color:{label_color}">{label_text}</div>
            <div class="prediction-prob">P(abnormal) = {prob:.1%}</div>
            <div class="prediction-prob" style="margin-top:0.4rem">
                Consecutive: <b>{cons}</b>/3
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ══════════════════════════════════════════════════════════════════
    # Row 2: BPM Gauge + HRV Metrics + Feature Table
    # ══════════════════════════════════════════════════════════════════
    col_bpm, col_hrv, col_feat = st.columns([1.2, 1.8, 1.8], gap="medium")

    bpm   = features.get("heart_rate", 0.0) if features else 0.0
    sdnn  = features.get("sdnn",       0.0) if features else 0.0
    rmssd = features.get("rmssd",      0.0) if features else 0.0
    rr_mean = features.get("rr_mean",  0.0) if features else 0.0

    with col_bpm:
        st.markdown('<div class="section-title">Heart Rate</div>', unsafe_allow_html=True)
        st.plotly_chart(make_bpm_gauge(bpm), use_container_width=True,
                        config={"displayModeBar": False})

    with col_hrv:
        st.markdown('<div class="section-title">HRV Metrics</div>', unsafe_allow_html=True)
        h1, h2, h3 = st.columns(3)
        cards = [
            (h1, "SDNN",    sdnn,   "ms",  "#A78BFA"),
            (h2, "RMSSD",   rmssd,  "ms",  "#34D399"),
            (h3, "RR Mean", rr_mean,"ms",  "#60A5FA"),
        ]
        for col, label_c, val, unit, color in cards:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label_c}</div>
                    <div class="metric-value" style="color:{color}">
                        {val:.1f}
                    </div>
                    <div class="metric-unit">{unit}</div>
                </div>
                """, unsafe_allow_html=True)

    with col_feat:
        st.markdown('<div class="section-title">Feature Values</div>', unsafe_allow_html=True)
        feature_display = [
            ("Heart Rate",    f"{features.get('heart_rate',0):.1f} BPM"),
            ("RR Mean",       f"{features.get('rr_mean',0):.1f} ms"),
            ("RR Std",        f"{features.get('rr_std',0):.1f} ms"),
            ("SDNN",          f"{features.get('sdnn',0):.1f} ms"),
            ("RMSSD",         f"{features.get('rmssd',0):.1f} ms"),
            ("Beat Variance", f"{features.get('beat_variance',0):.1f} ms²"),
            ("R-Peaks (5s)",  f"{features.get('r_peak_count',0)}"),
        ]
        rows_html = "".join([
            f'<div class="feature-row"><span class="feature-name">{n}</span>'
            f'<span class="feature-val">{v}</span></div>'
            for n, v in feature_display
        ])
        st.markdown(
            f'<div style="background:#0D1117;border:1px solid #1E293B;'
            f'border-radius:12px;padding:1rem">{rows_html}</div>',
            unsafe_allow_html=True
        )

    st.markdown("")

    # ══════════════════════════════════════════════════════════════════
    # Row 3: Alert History
    # ══════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Alert History (last 20 events)</div>',
                unsafe_allow_html=True)

    history = st.session_state.get("alert_history", [])
    if not history:
        st.markdown('<div style="color:#475569;font-size:0.85rem;padding:0.5rem 0">No events recorded yet.</div>',
                    unsafe_allow_html=True)
    else:
        for evt in reversed(history):
            cls = "alert-item" if evt["label"] == "ABNORMAL" else "alert-item-normal"
            st.markdown(
                f'<div class="{cls}">'
                f'{evt["time"]} &nbsp;·&nbsp; <b>{evt["label"]}</b>'
                f' &nbsp;·&nbsp; P={evt["prob"]:.1%}'
                f'</div>',
                unsafe_allow_html=True
            )

    # ══════════════════════════════════════════════════════════════════
    # Auto-Refresh
    # ══════════════════════════════════════════════════════════════════
    if running:
        interval_ms = st.session_state["refresh_interval"]
        st.markdown(
            f'<meta http-equiv="refresh" content="{interval_ms // 1000}">',
            unsafe_allow_html=True
        )
        time.sleep(interval_ms / 1000.0)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════

render_sidebar()
render_dashboard()
