"""
RailVision AI — SIH 1349
Minimalist Command-Center Interface
Ministry of Railways · CRIS · Integrated CCTV Intelligence
"""
import streamlit as st
import threading
import html as _html
import cv2
import numpy as np
import pandas as pd
import time
import datetime as dt
import tempfile
import base64
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO

st.set_page_config(
    page_title="RailVision — CRIS Command Center",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── AUTH GATE ─────────────────────────────────────────────────────────────────
# Set APP_PASSWORD in st.secrets (secrets.toml) or as an environment variable.
# If neither is configured the gate is skipped with a warning (safe for local dev).
import os as _os
import hmac as _hmac
try:
    _pwd_required = st.secrets.get("APP_PASSWORD", _os.environ.get("APP_PASSWORD", ""))
except Exception:
    # No secrets.toml present — fall back to environment variable only.
    _pwd_required = _os.environ.get("APP_PASSWORD", "")
if not _pwd_required:
    # No password configured — show a visible operator warning so this gap is never silent.
    st.sidebar.caption("⚠ No password configured — gate disabled")
else:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if not st.session_state["authenticated"]:
        # ── Login card ──────────────────────────────────────────────────────────
        # Inject scoped CSS once so the container renders as a centred card.
        # st.container(key=...) applies an "st-key-<key>" class to the wrapper
        # element, which is what the CSS below actually targets.
        st.markdown("""
        <style>
        div[data-testid="stVerticalBlockBorderWrapper"].st-key-rv_login_card {
            max-width: 340px;
            margin: 10vh auto 0;
        }
        div[data-testid="stVerticalBlockBorderWrapper"].st-key-rv_login_card
            > div[data-testid="stVerticalBlock"] {
            background: #141414;
            border: 1px solid #242424;
            border-radius: 4px;
            padding: 32px 28px 24px;
        }
        </style>""", unsafe_allow_html=True)

        # All login widgets (static header text + inputs) live inside this
        # container, so they are visually part of the same card.
        with st.container(key="rv_login_card", border=False):
            st.markdown("""
            <div style='font-size:0.60rem;letter-spacing:1.5px;text-transform:uppercase;
            color:#505050;margin-bottom:6px'>CRIS · Ministry of Railways</div>
            <div style='font-size:1.1rem;font-weight:600;color:#f0f0f0;
            margin-bottom:4px'>RailVision</div>
            <div style='font-size:0.70rem;color:#505050;margin-bottom:20px'>
            Authorised access only · Classification: Restricted</div>""",
            unsafe_allow_html=True)
            _entered = st.text_input("Access code", type="password",
                                     placeholder="Enter access code",
                                     label_visibility="collapsed")
            if st.button("Authenticate", type="primary", use_container_width=True):
                # Constant-time comparison — eliminates timing side-channel.
                if _hmac.compare_digest(_entered, _pwd_required):
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Invalid access code.")
        st.stop()


# ─── DESIGN TOKENS ────────────────────────────────────────────────────────────
# bg=#0c0c0c  surface=#141414  surface2=#1c1c1c  border=#242424
# text=#f0f0f0  text2=#a0a0a0  text3=#505050
# red=#dc2626  amber=#d97706  green=#16a34a  blue=#3b82f6  cyan=#0891b2

st.markdown("""
<style>
/* ── Root ── */
.stApp { background-color: #0c0c0c !important; }

section[data-testid="stSidebar"] {
    background-color: #0f0f0f !important;
    border-right: 1px solid #242424 !important;
}
.block-container {
    padding-top: 2.5rem !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ── Typography ── */
.stApp {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif !important;
    color: #f0f0f0 !important;
}
.stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3 {
    color: #f0f0f0 !important;
}
.mono {
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #a0a0a0 !important;
    font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #242424 !important;
    margin: 10px 0 !important;
}
section[data-testid="stSidebar"] > div {
    padding-bottom: 3rem !important;
}

/* ── System header ── */
.rv-header {
    background-color: #0f0f0f;
    border-bottom: 1px solid #242424;
    padding: 11px 20px;
}
.rv-header-table { width: 100%; border-collapse: collapse; }
.rv-header-table td { vertical-align: middle; padding: 0; }
.rv-header-org {
    font-size: 0.60rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #505050;
    margin-bottom: 2px;
}
.rv-header-name {
    font-size: 0.98rem;
    font-weight: 600;
    color: #f0f0f0;
    letter-spacing: -0.2px;
}
.rv-header-right {
    text-align: right;
}
.rv-header-clock {
    font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace;
    font-size: 0.78rem;
    color: #606060;
}
.rv-status-indicator {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background-color: #16a34a;
    vertical-align: middle;
    margin-right: 5px;
}
.rv-status-text {
    font-size: 0.65rem;
    color: #505050;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.rv-sih-tag {
    display: inline-block;
    font-size: 0.58rem;
    color: #505050;
    border: 1px solid #303030;
    padding: 2px 7px;
    letter-spacing: 0.8px;
    vertical-align: middle;
    margin-left: 8px;
    font-family: 'SF Mono', Consolas, monospace;
}

/* ── Nav tabs ── */
.rv-tabs {
    background-color: #0f0f0f;
    border-bottom: 1px solid #1e1e1e;
    padding: 0 20px;
    white-space: nowrap;
    overflow-x: auto;
}
.rv-tab {
    display: inline-block;
    padding: 9px 16px;
    font-size: 0.78rem;
    font-weight: 400;
    color: #505050;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    letter-spacing: 0.2px;
    text-decoration: none;
    transition: color 0.15s;
}
.rv-tab.on {
    color: #f0f0f0;
    font-weight: 500;
    border-bottom-color: #f0f0f0;
}
.rv-tab-right {
    float: right;
    font-family: 'SF Mono', Consolas, monospace;
    font-size: 0.65rem;
    color: #585858;
    padding-top: 11px;
}

/* ── Stat strip ── */
.rv-statstrip {
    background-color: #141414;
    border-bottom: 1px solid #1e1e1e;
    padding: 0;
    display: table;
    width: 100%;
    table-layout: fixed;
}
.rv-sc {
    display: table-cell;
    padding: 11px 20px;
    border-right: 1px solid #1e1e1e;
    vertical-align: middle;
}
.rv-sc:last-child { border-right: none; }
.rv-sc-val {
    font-size: 1.6rem;
    font-weight: 300;
    line-height: 1;
    font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace;
    letter-spacing: -1px;
    margin-bottom: 3px;
}
.rv-sc-lbl {
    font-size: 0.58rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #404040;
}
.col-w  { color: #f0f0f0; }
.col-r  { color: #dc2626; }
.col-a  { color: #d97706; }
.col-g  { color: #16a34a; }
.col-b  { color: #3b82f6; }
.col-c  { color: #0891b2; }
.col-m  { color: #707070; }

/* ── Content panels ── */
.rv-page { padding: 16px 20px; }

.rv-panel {
    background-color: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 3px;
    margin-bottom: 12px;
}
.rv-panel-hd {
    padding: 8px 12px;
    border-bottom: 1px solid #1e1e1e;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #505050;
    display: table;
    width: 100%;
    box-sizing: border-box;
}
.rv-panel-hd-right {
    display: table-cell;
    text-align: right;
    font-family: 'SF Mono', Consolas, monospace;
    color: #585858;
    font-weight: 400;
}
.rv-panel-bd { padding: 0; }
.rv-log-scroll {
    max-height: 420px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #2a2a2a #141414;
}
.rv-log-scroll::-webkit-scrollbar { width: 5px; }
.rv-log-scroll::-webkit-scrollbar-track { background: #141414; }
.rv-log-scroll::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 3px; }

/* ── Metric card ── */
.rv-metric {
    background-color: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 3px;
    padding: 14px 16px;
    border-top: 1px solid;
}
.rv-metric-val {
    font-size: 1.8rem;
    font-weight: 300;
    line-height: 1;
    margin-bottom: 5px;
    font-family: 'SF Mono', Consolas, monospace;
    letter-spacing: -0.5px;
}
.rv-metric-lbl {
    font-size: 0.60rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #404040;
}
.rv-metric.blue  { border-top-color: #1d4ed8; }
.rv-metric.blue  .rv-metric-val { color: #3b82f6; }
.rv-metric.cyan  { border-top-color: #0e7490; }
.rv-metric.cyan  .rv-metric-val { color: #0891b2; }
.rv-metric.red   { border-top-color: #991b1b; }
.rv-metric.red   .rv-metric-val { color: #dc2626; }
.rv-metric.amber { border-top-color: #92400e; }
.rv-metric.amber .rv-metric-val { color: #d97706; }
.rv-metric.green { border-top-color: #14532d; }
.rv-metric.green .rv-metric-val { color: #16a34a; }
.rv-metric.neutral{ border-top-color: #242424; }
.rv-metric.neutral .rv-metric-val { color: #707070; }

/* ── Status table ── */
.rv-stbl { width: 100%; border-collapse: collapse; }
.rv-stbl tr { border-bottom: 1px solid #1a1a1a; }
.rv-stbl tr:last-child { border-bottom: none; }
.rv-stbl td { padding: 8px 12px; font-size: 0.78rem; }
.rv-sk { color: #606060 !important; width: 48%; }
.rv-sv {
    font-family: 'SF Mono', Consolas, monospace !important;
    font-size: 0.74rem !important;
    color: #d0d0d0 !important;
    font-weight: 500 !important;
}
.rv-sv.g { color: #16a34a !important; }
.rv-sv.r { color: #dc2626 !important; }
.rv-sv.a { color: #d97706 !important; }
.rv-sv.b { color: #3b82f6 !important; }
.rv-sv.m { color: #505050 !important; }

/* ── Alert items ── */
.rv-alert {
    padding: 10px 12px;
    border-bottom: 1px solid #1a1a1a;
    display: table;
    width: 100%;
    box-sizing: border-box;
}
.rv-alert:last-child { border-bottom: none; }
.rv-alert-sev {
    display: table-cell;
    width: 70px;
    vertical-align: top;
    padding-top: 1px;
}
.rv-sev-badge {
    display: inline-block;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    padding: 2px 6px;
    border-radius: 2px;
    font-family: 'SF Mono', Consolas, monospace;
}
.sev-crit  { background: #1c0000; color: #dc2626; border: 1px solid #3d0000; }
.sev-high  { background: #1a0e00; color: #d97706; border: 1px solid #3d2000; }
.sev-med   { background: #0f0e00; color: #ca8a04; border: 1px solid #292500; }
.sev-low   { background: #001209; color: #16a34a; border: 1px solid #00280e; }
.rv-alert-body { display: table-cell; vertical-align: top; }
.rv-alert-evt { font-size: 0.78rem; font-weight: 500; color: #e8e8e8 !important; }
.rv-alert-det { font-size: 0.70rem; color: #606060 !important; margin-top: 2px; }
.rv-alert-meta {
    font-size: 0.62rem;
    color: #686868 !important;
    margin-top: 3px;
    font-family: 'SF Mono', Consolas, monospace;
}

/* ── Feed ── */
.rv-feed-wrap {
    background-color: #0a0a0a;
    border: 1px solid #1e1e1e;
    border-radius: 3px;
}
.rv-no-signal {
    padding: 70px 20px;
    text-align: center;
}
.rv-no-signal .ns-icon { font-size: 2rem; margin-bottom: 10px; color: #2a2a2a; }
.rv-no-signal .ns-title { font-size: 0.88rem; font-weight: 500; color: #3a3a3a; margin-bottom: 4px; }
.rv-no-signal .ns-sub { font-size: 0.72rem; color: #2a2a2a; }

/* ── Snapshot ── */
.rv-snap-meta {
    padding: 5px 9px;
    background-color: #141414;
    border-top: 1px solid #1e1e1e;
}
.rv-snap-evt { font-size: 0.66rem; font-weight: 600; color: #e0e0e0 !important; }
.rv-snap-inf { font-size: 0.60rem; color: #3d3d3d !important; font-family: 'SF Mono', Consolas, monospace; }

/* ── Banner ── */
.rv-banner-red {
    background-color: #0e0000;
    border: 1px solid #3d0000;
    border-radius: 3px;
    padding: 10px 16px;
    margin-bottom: 14px;
}
.rv-banner-blue {
    background-color: #00080e;
    border: 1px solid #0a2040;
    border-radius: 3px;
    padding: 10px 16px;
    margin-bottom: 14px;
}
.rv-banner-title-r { font-size: 0.95rem; font-weight: 600; color: #dc2626; }
.rv-banner-title-b { font-size: 0.95rem; font-weight: 600; color: #3b82f6; }
.rv-banner-sub { font-size: 0.70rem; color: #686868 !important; margin-top: 3px; }

/* ── Empty state ── */
.rv-empty {
    padding: 36px 16px;
    text-align: center;
    font-size: 0.76rem;
    color: #585858;
}

/* ── Streamlit native overrides ── */
.stButton > button {
    background-color: #1c1c1c !important;
    color: #a0a0a0 !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 3px !important;
    font-size: 0.82rem !important;
    padding: 9px 16px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}
.stButton > button[kind="primary"] {
    background-color: #1a1a1a !important;
    color: #f0f0f0 !important;
    border: 1px solid #3a3a3a !important;
}
.stButton > button:hover {
    border-color: #4a4a4a !important;
    color: #f0f0f0 !important;
}
div[data-testid="stFileUploader"] > div {
    background-color: #141414 !important;
    border: 1px dashed #2e2e2e !important;
    border-radius: 3px !important;
    padding: 16px !important;
}
div[data-testid="stFileUploader"] section {
    padding: 0 !important;
}
.stSelectbox > div > div {
    background-color: #141414 !important;
    border-color: #2e2e2e !important;
}
.stTextInput > div > div {
    background-color: #141414 !important;
    border-color: #2e2e2e !important;
}
div[data-testid="stDataFrame"] > div {
    background-color: #141414 !important;
    border: 1px solid #1e1e1e !important;
}
.stProgress > div > div > div {
    background-color: #dc2626 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in {
    "events": [], "total_alerts": 0, "last_alert": {},
    "history": [], "live": {"people": 0, "zone": 0, "risk": "—"},
    "snapshots": [], "stop_requested": False, "detection_warning": False,
    "last_vlm_call": 0.0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:14px 4px 10px'>
      <div style='font-size:0.60rem;letter-spacing:1.5px;text-transform:uppercase;color:#404040;margin-bottom:4px'>CRIS / Ministry of Railways</div>
      <div style='font-size:1.05rem;font-weight:600;color:#f0f0f0;letter-spacing:-0.3px'>RailVision</div>
      <div style='font-size:0.65rem;color:#404040;margin-top:1px'>Command Center · SIH-1349</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("<div style='font-size:0.58rem;letter-spacing:1.5px;text-transform:uppercase;color:#686868;padding:6px 0 6px'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("nav", [
        "Live Command Center",
        "Alerts & Security",
        "Station Analytics",
    ], label_visibility="collapsed")
    st.divider()

    st.markdown("<div style='font-size:0.58rem;letter-spacing:1.5px;text-transform:uppercase;color:#686868;padding:6px 0 6px'>Source</div>", unsafe_allow_html=True)
    source = st.radio("src", ["Upload Recording", "Live Webcam"], label_visibility="collapsed")
    st.divider()

    st.markdown("<div style='font-size:0.58rem;letter-spacing:1.5px;text-transform:uppercase;color:#686868;padding:6px 0 6px'>Detection</div>", unsafe_allow_html=True)
    model_opt   = st.selectbox("Model", [
        "yolo11n.pt   Nano",
        "yolo11s.pt   Small",
        "yolo11m.pt   Medium ✓",
    ], index=2)
    model_name  = model_opt.split()[0]
    cam_label   = st.text_input("Camera ID", "NDLS-P1-C04")
    conf_thresh = st.slider("Confidence",          0.05, 0.80, 0.15, 0.05)
    iou_thresh  = st.slider("IoU (crowd: high)",   0.20, 0.80, 0.65, 0.05)
    crowd_limit = st.slider("Crowd alert at",      2,    200,  15)
    skip_n      = st.slider("Frame skip",          1,    3,    1)
    st.divider()

    st.markdown("<div style='font-size:0.58rem;letter-spacing:1.5px;text-transform:uppercase;color:#686868;padding:6px 0 6px'>Modules</div>", unsafe_allow_html=True)
    mod_privacy    = st.checkbox("Face anonymisation",   True)
    mod_heatmap    = st.checkbox("Density heatmap",      False)
    mod_trails     = st.checkbox("Movement trails",      True)
    mod_restricted = st.checkbox("Restricted zone",      True)
    mod_track      = st.checkbox("Track intrusion",      True)
    mod_abandoned  = st.checkbox("Abandoned objects",    True)
    mod_behavior_ai = st.checkbox("AI Behavior Analysis (LLM)", True)
    mod_weapon     = st.checkbox("Weapon detection",     True)
    mod_staff      = st.checkbox("Staff post monitor",   True)
    st.divider()

    st.markdown("<div style='font-size:0.58rem;color:#585858;line-height:1.8;font-family:SF Mono,Consolas,monospace'>SYS: CRIS-ICIS-0392<br>ZONE: NORTHERN RAILWAY<br>ACCESS: RESTRICTED</div>", unsafe_allow_html=True)

# ─── AI MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_model(name):
    """Parse & cache YOLO weights once per process (shared read-only across sessions)."""
    return YOLO(name)

def _get_session_model(name):
    """Return a per-session YOLO wrapper stored in st.session_state.

    Each browser session gets its own YOLO Python object (and therefore its own
    Ultralytics predictor / tracker state), which means:
    - Resetting one session's tracker never touches another session's state.
    - Concurrent model.track() calls run on separate predictor objects — no
      shared-mutable-state race condition.

    The underlying weight tensors are still loaded only once (via get_model above)
    and are mmap-shared by the OS, so per-session memory overhead is modest
    (only the Python wrapper + predictor/tracker state duplicates, not the weights).
    """
    key = f"_model_{name}"
    if key not in st.session_state:
        # Warm the weight cache then hand a fresh wrapper to this session.
        try:
            get_model(name)           # ensures weights are parsed (cached)
            st.session_state[key] = YOLO(name)
        except Exception as e:
            st.error(f"Model error: {e}")
            st.stop()
    return st.session_state[key]

@st.cache_resource(show_spinner=False)
def get_face_cascade():
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return cc if not cc.empty() else None
    return None

_vlm_lock = threading.Lock()

@st.cache_resource(show_spinner="Loading behavior-analysis model…")
def get_vlm():
    """Load Moondream2 once per process. Shared read-only across sessions;
    calls are serialized via _vlm_lock since transformers .generate() is not
    safe to call concurrently on the same model instance from multiple threads."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = property(lambda self: {})
    model_id = "vikhyatk/moondream2"
    vlm = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    return vlm, tok

def describe_frame(frame_bgr, prompt):
    """Run one captioning/VQA call against a cropped or full BGR frame.
    Returns a short string description, or None on any failure (never raises)."""
    try:
        vlm, tok = get_vlm()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(rgb)
        with _vlm_lock:
            enc = vlm.encode_image(pil_img)
            answer = vlm.answer_question(enc, prompt, tok)
        return answer.strip()[:200] if answer else None
    except Exception:
        return None

def scan_for_abnormal_behavior(frame_bgr):
    """Ask the local VLM a general yes/no + description question about the whole frame.
    Returns (is_abnormal: bool, description: str) or (False, None) on any failure/parse issue."""
    raw = describe_frame(
        frame_bgr,
        "You are monitoring a railway station CCTV feed. Look for abnormal human behavior: "
        "fighting, running, falling, aggressive gestures, or any unusual activity. "
        "Respond in exactly this format: NORMAL or ABNORMAL: <one short sentence>."
    )
    if not raw:
        return False, None
    if raw.upper().startswith("ABNORMAL"):
        desc = raw.split(":", 1)[1].strip() if ":" in raw else raw
        return True, desc[:200]
    return False, None

def scan_for_fight(frame_bgr):
    """Ask VLM whether persons in the frame are fighting or physically struggling.
    Returns (is_fighting: bool, description: str) or (False, None) on failure."""
    raw = describe_frame(
        frame_bgr,
        "You are a railway station security system. Are people fighting, physically "
        "struggling, or striking each other in this image? "
        "Respond in exactly this format: FIGHTING: <one short sentence> or NOT FIGHTING."
    )
    if not raw:
        return False, None
    if raw.upper().startswith("FIGHTING"):
        desc = raw.split(":", 1)[1].strip() if ":" in raw else raw
        return True, desc[:200]
    return False, None

def scan_for_pickpocket(frame_bgr):
    """Ask VLM whether a pickpocketing or bag-snatching event is visible.
    Returns (is_theft: bool, description: str) or (False, None) on failure."""
    raw = describe_frame(
        frame_bgr,
        "You are a railway station security system. Is anyone grabbing, snatching, or "
        "taking a bag, wallet, or item from another person in this image? "
        "Respond in exactly this format: THEFT: <one short sentence> or NO THEFT."
    )
    if not raw:
        return False, None
    if raw.upper().startswith("THEFT"):
        desc = raw.split(":", 1)[1].strip() if ":" in raw else raw
        return True, desc[:200]
    return False, None

def scan_for_fallen_person(frame_bgr):
    """Ask VLM whether a person has fallen or collapsed in the frame crop.
    Returns (is_fallen: bool, description: str) or (False, None) on failure."""
    raw = describe_frame(
        frame_bgr,
        "You are a railway station safety system. Is there a person who has fallen down "
        "or collapsed on the ground in this image? "
        "Respond in exactly this format: FALLEN: <one short sentence> or NOT FALLEN."
    )
    if not raw:
        return False, None
    if raw.upper().startswith("FALLEN"):
        desc = raw.split(":", 1)[1].strip() if ":" in raw else raw
        return True, desc[:200]
    return False, None

def scan_for_package_drop(crop_bgr):
    """Ask VLM whether the cropped region shows a bag/package that was left unattended.
    Returns (is_suspicious: bool, description: str) or (False, None) on failure."""
    raw = describe_frame(
        crop_bgr,
        "You are a railway station security system. Is this a bag, luggage, or package "
        "that appears to have been left unattended or dropped intentionally? "
        "Respond in exactly this format: SUSPICIOUS: <one short sentence> or NOT SUSPICIOUS."
    )
    if not raw:
        return False, None
    if raw.upper().startswith("SUSPICIOUS"):
        desc = raw.split(":", 1)[1].strip() if ":" in raw else raw
        return True, desc[:200]
    return False, None

def scan_for_unattended_child(frame_bgr):
    """Ask VLM whether a young child appears to be alone/unattended in the frame.
    Returns (is_child_alone: bool, description: str) or (False, None) on failure."""
    raw = describe_frame(
        frame_bgr,
        "You are a railway station safety system. Is there a young child (under ~12 years) "
        "who appears to be alone without a parent or guardian nearby? "
        "Respond in exactly this format: CHILD ALONE: <one short sentence> or NO CHILD ALONE."
    )
    if not raw:
        return False, None
    if raw.upper().startswith("CHILD ALONE") or raw.upper().startswith("CHILD_ALONE"):
        desc = raw.split(":", 1)[1].strip() if ":" in raw else raw
        return True, desc[:200]
    return False, None

try:
    model = _get_session_model(model_name)
except Exception as e:
    st.error(f"Model error: {e}"); st.stop()
face_cc    = get_face_cascade()
THREATS    = {"knife", "baseball bat", "scissors", "gun"}
PERSON_CLS = 0

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _ts():  return dt.datetime.now().strftime("%H:%M:%S")
def _dts(): return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def seed_demo_events():
    if st.session_state.get("_demo_seeded"): return
    st.session_state["_demo_seeded"] = True
    if st.session_state["events"]: return  # never overwrite real data
    demo = [
        ("CROWD", "Overcrowding", "HIGH", "NDLS-P1-C04", "18 in zone (limit 15)"),
        ("SECURITY", "Restricted Zone Intrusion", "HIGH", "NDLS-P3-C02", "1 person(s)"),
        ("OPERATIONS", "Staff Post Unattended", "MEDIUM", "NDLS-P2-C01", "Post vacant >7s"),
        ("SECURITY", "Weapon Detected", "CRITICAL", "NDLS-P1-C04", "Detected: knife"),
    ]
    for cat, evt, sev, cam, detail in demo:
        add_event(cat, evt, sev, cam, detail, snap_b64=None)

def add_event(cat, evt, sev, cam, detail, snap_b64=None):
    key = f"{cat}|{evt}|{cam}"; now = time.time()
    deb = 2 if sev == "CRITICAL" else 5
    last = st.session_state["last_alert"]
    if key in last and now - last[key] < deb: return False
    last[key] = now
    sid = f"snap_{int(now*1000)}"
    st.session_state["events"].insert(0, {
        "Time": _dts(), "Cat": cat, "Camera": cam,
        "Event": evt, "Severity": sev, "Details": detail,
        "SnapID": sid if snap_b64 else "",
    })
    st.session_state["events"] = st.session_state["events"][:500]
    st.session_state["total_alerts"] += 1
    if snap_b64:
        st.session_state["snapshots"].insert(0, {
            "id": sid, "cat": cat, "event": evt,
            "sev": sev, "cam": cam, "time": _dts(), "img": snap_b64,
        })
        st.session_state["snapshots"] = st.session_state["snapshots"][:60]
    return True

seed_demo_events()

def forecast_zone_count(hist, horizon_min=10, window=20):
    """
    Simple linear-trend forecast of 'zone' occupancy from recent history.
    hist: list of {"t","people","zone"} dicts, chronological order (oldest first).
    Returns (projected_value:int, trend:str) where trend is 'rising','falling','steady'.
    Returns (None, None) if there isn't enough data to forecast.
    """
    if len(hist) < 5:
        return None, None
    recent = hist[-window:] if len(hist) > window else hist
    ys = np.array([h["zone"] for h in recent], dtype=float)
    xs = np.arange(len(ys), dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    projected = intercept + slope * (len(ys) - 1 + horizon_min)
    projected = max(0, round(projected))
    if abs(slope) < 0.02:
        trend = "steady"
    elif slope > 0:
        trend = "rising"
    else:
        trend = "falling"
    return projected, trend

def pt_in(cx, cy, b): x1,y1,x2,y2=b; return x1<=cx<=x2 and y1<=cy<=y2

def draw_zone(img, box, label, color, alpha=0.06):
    x1,y1,x2,y2=box; H,W=img.shape[:2]
    x1,y1,x2,y2=max(0,x1),max(0,y1),min(W-1,x2),min(H-1,y2)
    ov=img.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)
    cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
    tw,th=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)[0]
    cv2.rectangle(img,(x1,y1-th-4),(x1+tw+6,y1),color,-1)
    cv2.putText(img,label,(x1+3,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1,cv2.LINE_AA)

def highlight_zone(img, box, label, sev):
    C={"CRITICAL":(180,0,0),"HIGH":(30,100,200),"MEDIUM":(0,130,190),"LOW":(0,140,40)}
    col=C.get(sev,(160,120,0))
    x1,y1,x2,y2=box; H,W=img.shape[:2]
    x1,y1,x2,y2=max(0,x1),max(0,y1),min(W-1,x2),min(H-1,y2)
    ov=img.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),col,-1)
    cv2.addWeighted(ov,0.16,img,0.84,0,img)
    cv2.rectangle(img,(x1,y1),(x2,y2),col,2)
    arm,t=16,3
    for (cx,cy,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img,(cx,cy),(cx+dx*arm,cy),col,t,cv2.LINE_AA)
        cv2.line(img,(cx,cy),(cx,cy+dy*arm),col,t,cv2.LINE_AA)
    bh=20; by1=max(0,y2-bh); ov2=img.copy()
    cv2.rectangle(ov2,(x1,by1),(x2,y2),col,-1)
    cv2.addWeighted(ov2,0.80,img,0.20,0,img)
    tag=f"[{sev}] {label}"
    (tw,_),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
    cv2.putText(img,tag,(x1+max(0,((x2-x1)-tw)//2),by1+14),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1,cv2.LINE_AA)

def snap(frame):
    h,w=frame.shape[:2]
    if w>480: s=480/w; frame=cv2.resize(frame,(480,int(h*s)),interpolation=cv2.INTER_AREA)
    ok,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,76])
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else None

def detect_all(frame, conf, iou):
    kw=dict(source=frame,conf=conf,iou=iou,imgsz=1280,verbose=False)
    tr="custom_tracker.yaml" if Path("custom_tracker.yaml").exists() else "bytetrack.yaml"
    try: return model.track(persist=True,tracker=tr,**kw)[0], True
    except Exception: pass
    try: return model.predict(**kw)[0], False
    except Exception:
        st.session_state["detection_warning"] = True
        return None, False

def blur_face(frame, x1, y1, x2, y2):
    crop=frame[y1:y2,x1:x2]
    if crop.size==0: return 0
    if face_cc is not None:
        g=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        fs=face_cc.detectMultiScale(g,1.1,4,minSize=(16,16))
        for (fx,fy,fw,fh) in fs:
            fa1,fa2=max(0,y1+fy),min(frame.shape[0],y1+fy+fh)
            fb1,fb2=max(0,x1+fx),min(frame.shape[1],x1+fx+fw)
            roi=frame[fa1:fa2,fb1:fb2]
            if roi.size: frame[fa1:fa2,fb1:fb2]=cv2.GaussianBlur(roi,(31,31),0)
        return len(fs)
    hy=y1+max(6,int((y2-y1)*0.28)); roi=frame[y1:hy,x1:x2]
    if roi.size: frame[y1:hy,x1:x2]=cv2.GaussianBlur(roi,(31,31),0)
    return 1

# ─── UI COMPONENTS ────────────────────────────────────────────────────────────
def header_html():
    alerts = st.session_state["total_alerts"]
    dot    = "background-color:#dc2626" if alerts > 0 else "background-color:#16a34a"
    status = f"<span style='{dot};width:6px;height:6px;border-radius:50%;display:inline-block;vertical-align:middle;margin-right:5px'></span><span style='font-size:0.63rem;color:#484848;letter-spacing:0.8px;text-transform:uppercase'>System Online</span>"
    return f"""<div class='rv-header'>
      <table class='rv-header-table'><tr>
        <td style='width:28px;padding-right:10px;font-size:1.4rem'>🎯</td>
        <td>
          <div class='rv-header-org'>Ministry of Railways · CRIS · Integrated CCTV Intelligence</div>
          <div class='rv-header-name'>RailVision Command Center
            <span class='rv-sih-tag'>SIH-1349</span>
          </div>
        </td>
        <td style='text-align:right;white-space:nowrap;padding-left:12px'>
          <div class='rv-header-clock'>{dt.datetime.now().strftime('%d %b %Y  ·  %H:%M:%S')}</div>
          <div style='margin-top:4px'>{status}</div>
        </td>
      </tr></table>
    </div>"""

def tabs_html(active):
    defs = [
        ("Live Command Center", "Live Cameras"),
        ("Alerts & Security",   "Alerts"),
        ("Station Analytics",   "Analytics"),
    ]
    t = "".join(
        f'<span class="rv-tab{"  on" if p == active else ""}">{lbl}</span>'
        for p, lbl in defs
    )
    _cam_safe = _html.escape(cam_label)
    return f"""<div class='rv-tabs'>{t}
      <span class='rv-tab-right'>{_cam_safe} &nbsp;·&nbsp; {dt.datetime.now().strftime('%H:%M')}</span>
    </div>"""

def statstrip_html(p, z, r, a, fps, faces):
    rc = {"CRITICAL":"col-r","WARNING":"col-a","NORMAL":"col-g","—":"col-m","COMPLETE":"col-b"}.get(r,"col-m")
    def sc(val, lbl, cls):
        return f'<div class="rv-sc"><div class="rv-sc-val {cls}">{val}</div><div class="rv-sc-lbl">{lbl}</div></div>'
    return f"""<div class='rv-statstrip'>
      {sc(p,   'People',       'col-w')}
      {sc(z,   'In Zone',      'col-c')}
      {sc(r,   'Status',       rc)}
      {sc(a,   'Alerts',       'col-r' if a > 0 else 'col-m')}
      {sc(f'{fps:.0f}', 'FPS', 'col-m')}
      {sc(faces,'Blurred',     'col-m')}
    </div>"""

def metric_html(val, lbl, cls):
    return f'<div class="rv-metric {cls}"><div class="rv-metric-val">{val}</div><div class="rv-metric-lbl">{lbl}</div></div>'

def panel(title, body_html, right=None):
    rh = f'<span class="rv-panel-hd-right">{right}</span>' if right else ""
    return f"""<div class='rv-panel'>
      <div class='rv-panel-hd'><span>{title}</span>{rh}</div>
      <div class='rv-panel-bd'>{body_html}</div>
    </div>"""

def status_rows(rows):
    inner = "".join(
        f'<tr><td class="rv-sk">{k}</td><td class="rv-sv {cls}">{v}</td></tr>'
        for k, v, cls in rows
    )
    return f'<table class="rv-stbl">{inner}</table>'

def alerts_html(events, n=10):
    cat_colors = {
        "CROWD":      "#0891b2",
        "SECURITY":   "#dc2626",
        "OPERATIONS": "#d97706",
        "BEHAVIOR":   "#8a5fd6",
    }
    sc = {"CRITICAL":"crit","HIGH":"high","MEDIUM":"med","LOW":"low"}
    rows = ""
    for e in events[:n]:
        c = sc.get(e.get("Severity",""), "low")
        _evt  = _html.escape(str(e.get('Event','')))
        _det  = _html.escape(str(e.get('Details',''))[:72])
        _cam  = _html.escape(str(e.get('Camera','')))
        rows += f"""<div class='rv-alert'>
          <div class='rv-alert-sev'><span class='rv-sev-badge sev-{c}'>{e.get('Severity','')}</span></div>
          <div class='rv-alert-body'>
            <div class='rv-alert-evt'>{_evt}</div>
            <div class='rv-alert-det'>{_det}</div>
            <div class='rv-alert-meta'>{e.get('Time','')} · {_cam}</div>
          </div>
        </div>"""
    if not rows:
        rows = "<div class='rv-empty'>No incidents recorded.</div>"
    return rows

def snap_gallery(snaps, n=9):
    sc = {"CRITICAL":"#dc2626","HIGH":"#d97706","MEDIUM":"#ca8a04","LOW":"#16a34a"}
    if not snaps:
        st.markdown("<div class='rv-empty'>No snapshots yet — captured automatically on alerts.</div>", unsafe_allow_html=True)
        return
    for i in range(0, min(len(snaps), n), 3):
        row  = snaps[i:i+3]
        cols = st.columns(3, gap="small")
        for col, s in zip(cols, row):
            bc = sc.get(s["sev"], "#505050")
            with col:
                st.image(base64.b64decode(s["img"]), width="stretch")
                st.markdown(
                    f"<div style='background:#141414;border-left:2px solid {bc};"
                    f"padding:4px 8px;margin-top:-4px'>"
                    f"<div style='font-size:0.65rem;font-weight:600;color:{bc};"
                    f"font-family:SF Mono,Consolas,monospace'>[{s['sev']}] {s['event']}</div>"
                    f"<div style='font-size:0.60rem;color:#404040;font-family:SF Mono,Consolas,monospace'>"
                    f"{s['time']} · {s['cam']}</div></div>",
                    unsafe_allow_html=True)

# ─── GLOBAL HEADER ────────────────────────────────────────────────────────────
st.markdown(header_html(), unsafe_allow_html=True)
st.markdown(tabs_html(page), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ALERTS & SECURITY
# ══════════════════════════════════════════════════════════════════════════════
if page == "Alerts & Security":
    st.markdown("<div style='padding:16px 20px 0'>", unsafe_allow_html=True)
    st.markdown("""<div class='rv-banner-red'>
      <div class='rv-banner-title-r'>RPF Security Operations</div>
      <div class='rv-banner-sub'>Restricted · Authorised RPF personnel only · Classification: Sensitive</div>
    </div>""", unsafe_allow_html=True)

    all_ev = st.session_state["events"]
    sec_ev = [e for e in all_ev if e["Cat"] == "SECURITY"]
    wep_ev = [e for e in sec_ev if "Weapon"     in e["Event"]]
    rze_ev = [e for e in sec_ev if "Restricted" in e["Event"]]
    obj_ev = [e for e in sec_ev if "Unattended" in e["Event"]]

    k1,k2,k3,k4 = st.columns(4, gap="small")
    k1.markdown(metric_html(len(sec_ev),  "Security Alerts",  "red"),    unsafe_allow_html=True)
    k2.markdown(metric_html(len(wep_ev),  "Weapon Alerts",    "amber"),  unsafe_allow_html=True)
    k3.markdown(metric_html(len(rze_ev),  "Zone Intrusions",  "neutral"),unsafe_allow_html=True)
    k4.markdown(metric_html(len(obj_ev),  "Abandoned Objects","neutral"),unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 1], gap="small")

    with left:
        st.markdown(panel("Incident Log", f"<div class='rv-log-scroll'>{alerts_html(sec_ev, 20)}</div>", right=str(len(sec_ev))),
                    unsafe_allow_html=True)
        if sec_ev:
            df = pd.DataFrame(sec_ev).drop(columns=["Cat","SnapID"], errors="ignore")
            st.download_button("Export CSV",
                df.to_csv(index=False).encode(),
                f"rpf_{dt.date.today()}.csv", "text/csv")

    with right:
        sec_snaps = [s for s in st.session_state["snapshots"] if s["cat"]=="SECURITY"]
        st.markdown('<div class="rv-panel"><div class="rv-panel-hd">Evidence Snapshots</div><div class="rv-panel-bd">', unsafe_allow_html=True)
        snap_gallery(sec_snaps)
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STATION ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Station Analytics":
    st.markdown("<div style='padding:16px 20px 0'>", unsafe_allow_html=True)
    st.markdown("""<div class='rv-banner-blue'>
      <div class='rv-banner-title-b'>Station Operations Dashboard</div>
      <div class='rv-banner-sub'>Crowd management · Staff monitoring · Operational intelligence</div>
    </div>""", unsafe_allow_html=True)

    live   = st.session_state["live"]
    all_ev = st.session_state["events"]
    c_ev   = [e for e in all_ev if e["Cat"]=="CROWD"]
    o_ev   = [e for e in all_ev if e["Cat"]=="OPERATIONS"]
    risk   = live.get("risk","—")
    rm     = {"CRITICAL":"red","WARNING":"amber","NORMAL":"green"}.get(risk,"neutral")

    k1,k2,k3,k4 = st.columns(4, gap="small")
    k1.markdown(metric_html(live["people"],      "People on Platform",  "blue"),  unsafe_allow_html=True)
    k2.markdown(metric_html(live["zone"],        "In Crowd Zone",       "cyan"),  unsafe_allow_html=True)
    k3.markdown(metric_html(risk,                "Crowd Status",        rm),      unsafe_allow_html=True)
    k4.markdown(metric_html(len(c_ev)+len(o_ev),"Operational Alerts",  "amber"), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    tl, tr = st.columns([1.6, 1], gap="small")

    with tl:
        st.markdown('<div class="rv-panel"><div class="rv-panel-hd">Crowd Density — Time Series</div><div class="rv-panel-bd" style="padding:8px">', unsafe_allow_html=True)
        hist = st.session_state["history"]
        if hist:
            df_h = pd.DataFrame(hist).set_index("t")[["people","zone"]]
            df_h.columns=["Total People","In Crowd Zone"]
            st.line_chart(df_h, height=190, width="stretch")
        else:
            st.markdown("<div class='rv-empty'>Awaiting data from Live Command Center.</div>", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="rv-panel"><div class="rv-panel-hd">Projected Crowd Trend (next 10 min)</div><div class="rv-panel-bd" style="padding:8px">', unsafe_allow_html=True)
        proj, trend = forecast_zone_count(hist)
        if proj is None:
            st.markdown("<div class='rv-empty'>Not enough data yet to forecast.</div>", unsafe_allow_html=True)
        else:
            trend_color = {"rising": "#b03030", "falling": "#2f7a3d", "steady": "#585858"}.get(trend, "#585858")
            trend_label = {"rising": "▲ Rising", "falling": "▼ Falling", "steady": "● Steady"}.get(trend, trend)
            st.markdown(
                f"<div style='font-size:1.6rem;font-weight:600;color:#f0f0f0'>{proj} <span style='font-size:0.8rem;font-weight:400;color:#585858'>people in zone (est.)</span></div>"
                f"<div style='font-size:0.75rem;color:{trend_color};margin-top:4px'>{trend_label}</div>"
                f"<div style='font-size:0.65rem;color:#585858;margin-top:8px'>Linear projection from the last {min(len(hist),20)} readings. Indicative only — not a guarantee.</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div></div>", unsafe_allow_html=True)

    with tr:
        combined = c_ev + o_ev
        st.markdown(panel("Operational Alerts", alerts_html(combined, 10), right=str(len(combined))),
                    unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="rv-panel"><div class="rv-panel-hd">Evidence Snapshots — Operational</div><div class="rv-panel-bd">', unsafe_allow_html=True)
    ops_snaps = [s for s in st.session_state["snapshots"] if s["cat"] in ("CROWD","OPERATIONS")]
    snap_gallery(ops_snaps)
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LIVE COMMAND CENTER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Command Center":

    stat_slot = st.empty()
    stat_slot.markdown(statstrip_html(0,0,"—",0,0.0,0), unsafe_allow_html=True)

    st.markdown("<div style='padding:10px 20px 0'>", unsafe_allow_html=True)
    feed_col, info_col = st.columns([2.6, 1], gap="small")

    with feed_col:
        st.markdown('<div class="rv-panel"><div class="rv-panel-hd">Live CCTV — ' + _html.escape(cam_label) + '</div>', unsafe_allow_html=True)
        frame_slot = st.empty()
        frame_slot.markdown("""<div class='rv-feed-wrap'><div class='rv-no-signal'>
          <div class='ns-icon'>⬛</div>
          <div class='ns-title'>No Signal</div>
          <div class='ns-sub'>Upload a recording or connect webcam, then press Start</div>
        </div></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with info_col:
        st.markdown('<div class="rv-panel"><div class="rv-panel-hd">Station Status</div>', unsafe_allow_html=True)
        status_slot = st.empty()
        status_slot.markdown(status_rows([
            ("Threat Status",    "CLEAR",   "g"),
            ("Crowd Status",     "STANDBY", "m"),
            ("Zone Count",       "—",       ""),
            ("Total Persons",    "—",       ""),
            ("Staff Post",       "—",       ""),
            ("Faces Blurred",    "—",       ""),
            ("AI Engine",        "IDLE",    "m"),
        ]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="rv-panel"><div class="rv-panel-hd">Incident Log</div><div class="rv-log-scroll">', unsafe_allow_html=True)
        incident_slot = st.empty()
        incident_slot.markdown("<div class='rv-empty'>No incidents.</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Upload & start
    upload = None
    if source == "Upload Recording":
        upload = st.file_uploader("CCTV footage", type=["mp4","avi","mov","mkv"],
                                   label_visibility="collapsed")

    start = st.button("▶  Start AI Processing", type="primary", width="stretch")
    if st.button("■ Stop", key="btn_stop"):
        st.session_state["stop_requested"] = True
    st.markdown("</div>", unsafe_allow_html=True)

    if start:
        if source == "Upload Recording" and not upload:
            st.warning("No video selected."); st.stop()

        st.session_state["stop_requested"] = False
        st.session_state["detection_warning"] = False
        # Each session owns its own YOLO instance (_get_session_model), so
        # tracker state is already isolated. We simply reset the predictor on
        # this session's private model object to clear IDs from a previous run
        # in the same browser tab — harmless and never touches other sessions.
        try:
            model.predictor = None
        except Exception:
            pass

        tmp_path = None
        if source == "Upload Recording":
            suf=Path(upload.name).suffix or ".mp4"
            tmp=tempfile.NamedTemporaryFile(delete=False, suffix=suf)
            tmp.write(upload.getbuffer()); tmp.close()
            tmp_path=tmp.name; cap=cv2.VideoCapture(tmp_path)
        else:
            cap=cv2.VideoCapture(0)

        if not cap.isOpened(): st.error("Cannot open source."); st.stop()

        src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        prog     = st.progress(0, "Processing…") if total_fr > 0 else None
        warn_slot = st.empty()

        heat=deque(maxlen=10000); obj_tracks=defaultdict(lambda:{"first":0,"seen":0})
        trails=defaultdict(lambda:deque(maxlen=45))
        dwell_start=defaultdict(lambda: None)   # tid -> frame_idx when this track was first seen stationary
        prev_gray_frame = None
        last_general_scan = 0.0
        last_fight_scan   = 0.0
        last_pick_scan    = 0.0
        last_fall_scan    = 0.0
        last_drop_scan    = 0.0
        last_child_scan   = 0.0
        prev_pcents       = []   # person centres from previous frame (for package-drop proximity check)
        pkg_close_on_drop = {}   # obj tid -> bool: was a person nearby when the object first appeared?
        count_history=deque(maxlen=5)  # Temporal smoothing buffer
        frame_idx=0; staff_empty=0; t_prev=time.time()
        proc_fps=0.0; faces_total=0; person_count=0; zone_count=0; risk="—"
        st.session_state["history"]=[]
        HUD=72

        try:
          while cap.isOpened():
            if st.session_state.get("stop_requested", False):
                break
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if skip_n > 1 and frame_idx % skip_n != 0: continue

            h0,w0=frame.shape[:2]
            if w0>1280: s=1280/w0; frame=cv2.resize(frame,(1280,int(h0*s)),interpolation=cv2.INTER_LINEAR)
            H,W=frame.shape[:2]

            gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_score = 0.0
            if prev_gray_frame is not None and prev_gray_frame.shape == gray_now.shape:
                diff = cv2.absdiff(gray_now, prev_gray_frame)
                motion_score = float((diff > 30).sum()) / diff.size  # fraction of pixels that changed significantly
            prev_gray_frame = gray_now

            crowd_zone=(int(W*0.05),HUD+int((H-HUD)*0.04),int(W*0.95),int(H*0.92))
            restr_zone=(int(W*0.02),HUD,int(W*0.28),HUD+int((H-HUD)*0.42))
            staff_zone=(int(W*0.68),int(H*0.60),int(W*0.97),int(H*0.95))
            track_zone=(int(W*0.02), int(H*0.90), int(W*0.98), int(H*0.99))

            result,has_ids=detect_all(frame,conf_thresh,iou_thresh)
            if st.session_state.get("detection_warning", False):
                warn_slot.warning("⚠ Detection failed on this frame — retrying next frame.")
                st.session_state["detection_warning"] = False
            else:
                warn_slot.empty()
            person_count=0; zone_count=0; staff_count=0; faces_blurred=0; pcents=[]
            threat_hits=[]

            if result is not None and result.boxes is not None and len(result.boxes):
                nm=model.names
                for b in result.boxes:
                    cid=int(b.cls[0])
                    conf=float(b.conf[0])
                    lbl=nm.get(cid,"")
                    
                    if cid==PERSON_CLS:
                        x1,y1,x2,y2=map(int,b.xyxy[0].tolist())
                        x1,y1=max(0,x1),max(0,y1); x2,y2=min(W-1,x2),min(H-1,y2)
                        w_box, h_box = x2 - x1, y2 - y1
                        # SIZE FILTER: Reject impossibly small artifacts (<15px) or massive screen-filling glitches.
                        # Exempt boxes touching the bottom frame edge — this indicates a person standing
                        # close to the camera (common in Live Webcam mode) rather than a detection artifact.
                        touches_bottom = y2 >= H - 5
                        if w_box < 15 or h_box < 30 or w_box > W*0.6 or (h_box > H*0.85 and not touches_bottom): continue
                        if x2<=x1 or y2<=y1: continue
                        cx,cy=(x1+x2)//2,(y1+y2)//2
                        tid=int(b.id[0]) if (has_ids and b.id is not None and len(b.id)>0) else None
                        person_count+=1; heat.append((cx,cy)); pcents.append((cx,cy))
                        if pt_in(cx,cy,crowd_zone): zone_count+=1
                        if pt_in(cx,cy,staff_zone): staff_count+=1
                        if mod_trails and tid is not None:
                            tr=trails[tid]; tr.append((cx,cy))
                            if len(tr)>1:
                                pts=np.array(tr,dtype=np.int32).reshape(-1,1,2)
                                cv2.polylines(frame,[pts],False,(50,150,70),1,cv2.LINE_AA)
                        if mod_behavior_ai and tid is not None:
                            tr_hist = trails.get(tid)
                            if tr_hist and len(tr_hist) >= 10:
                                pts = list(tr_hist)[-10:]
                                disp = max(((px-pts[0][0])**2+(py-pts[0][1])**2)**0.5 for px,py in pts)
                                if disp < 25:  # minimal movement over last 10 tracked positions
                                    if dwell_start[tid] is None: dwell_start[tid] = frame_idx
                                    dwell_sec = (frame_idx - dwell_start[tid]) / src_fps
                                    if dwell_sec >= 25 and (time.time() - st.session_state["last_vlm_call"]) >= 8:
                                        st.session_state["last_vlm_call"] = time.time()
                                        desc = describe_frame(frame[max(0,y1):y2, max(0,x1):x2],
                                                               "Describe this person's posture and behavior in one short sentence.")
                                        detail = desc if desc else f"Person stationary for {dwell_sec:.0f}s"
                                        highlight_zone(frame,(x1,y1,x2,y2),"PROLONGED LOITERING","MEDIUM")
                                        s=snap(frame); add_event("BEHAVIOR","Prolonged Loitering","MEDIUM",cam_label,detail,s)
                                        dwell_start[tid] = frame_idx  # reset so it doesn't refire every frame
                                else:
                                    dwell_start[tid] = None
                        if mod_privacy: faces_blurred+=blur_face(frame,x1,y1,x2,y2)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,165,70),1)
                        if tid is not None:
                            cv2.putText(frame,f"P{tid}",(x1+2,min(y2-2,y1+11)),cv2.FONT_HERSHEY_SIMPLEX,0.28,(0,180,80),1,cv2.LINE_AA)
                            
                    elif lbl in THREATS and mod_weapon and conf>=max(conf_thresh,0.45):
                        threat_hits.append((lbl, b.xyxy[0].tolist()))
                        
                    elif mod_abandoned and lbl not in THREATS and frame_idx>int(src_fps*8):
                        tid=int(b.id[0]) if (has_ids and b.id is not None and len(b.id)>0) else None
                        if tid is not None:
                            rec=obj_tracks[tid]
                            if rec["first"]==0: rec["first"]=frame_idx
                            rec["seen"]+=1
                            bx1,by1,bx2,by2=map(int,b.xyxy[0].tolist())
                            ocx,ocy=(bx1+bx2)//2,(by1+by2)//2
                            age=(frame_idx-rec["first"])/src_fps
                            nearest=min((((ocx-px)**2+(ocy-py)**2)**0.5 for px,py in pcents),default=9999)
                            if age>=10 and nearest>130:
                                highlight_zone(frame,(bx1,by1,bx2,by2),"UNATTENDED OBJECT","HIGH")
                                s=snap(frame); add_event("SECURITY","Unattended Object","HIGH",cam_label,f"'{lbl}' static {age:.0f}s",s)

            # TEMPORAL SMOOTHING: Rolling average over last 5 frames to eliminate flickering numbers
            count_history.append((person_count, zone_count))
            person_count = int(np.mean([x[0] for x in count_history]))
            zone_count = int(np.mean([x[1] for x in count_history]))

            faces_total += faces_blurred

            # Threats rendering
            for (lbl,coords) in threat_hits:
                tx1,ty1,tx2,ty2=map(int,coords)
                highlight_zone(frame,(tx1,ty1,tx2,ty2),f"WEAPON:{lbl.upper()}","CRITICAL")
                s=snap(frame); add_event("SECURITY","Weapon Detected","CRITICAL",cam_label,f"Detected: {lbl}",s)

            # Heatmap
            if mod_heatmap and len(heat)>5:
                ov=np.zeros((H,W),dtype=np.uint8)
                for px,py in heat:
                    if 0<=px<W and 0<=py<H: cv2.circle(ov,(px,py),20,200,-1)
                ov=cv2.GaussianBlur(ov,(0,0),22)
                hc=cv2.applyColorMap(ov,cv2.COLORMAP_JET)
                msk=ov>10; frame[msk]=cv2.addWeighted(frame,0.52,hc,0.48,0)[msk]

            draw_zone(frame,crowd_zone,"CROWD ZONE",(180,130,0))
            if mod_staff:      draw_zone(frame,staff_zone,"STAFF POST",(0,140,190))
            if mod_restricted: draw_zone(frame,restr_zone,"RESTRICTED",(150,0,0))
            if mod_track:      draw_zone(frame,track_zone,"TRACK ZONE",(0,0,150))

            if mod_behavior_ai and len(pcents) >= 2 and (time.time() - st.session_state["last_vlm_call"]) >= 8:
                close_pairs = []
                for i in range(len(pcents)):
                    for j in range(i+1, len(pcents)):
                        d = ((pcents[i][0]-pcents[j][0])**2 + (pcents[i][1]-pcents[j][1])**2) ** 0.5
                        if d < 60:
                            close_pairs.append((i, j))
                if not hasattr(describe_frame, "_prev_close_pairs"):
                    describe_frame._prev_close_pairs = set()
                prev = describe_frame._prev_close_pairs
                curr = set(close_pairs)
                separated = prev - curr
                if separated:
                    st.session_state["last_vlm_call"] = time.time()
                    desc = describe_frame(frame, "Two people who were close together just separated. Describe what is visible in one short sentence, focusing on any bag, object, or hand movement.")
                    detail = desc if desc else "Two persons in close proximity separated abruptly"
                    highlight_zone(frame,(int(W*0.05),HUD,int(W*0.95),H-1),"BEHAVIOR ALERT — REVIEW ADVISED","MEDIUM")
                    s=snap(frame); add_event("BEHAVIOR","Possible Unauthorized Item Removal — Needs Review","MEDIUM",cam_label,detail,s)
                describe_frame._prev_close_pairs = curr

            if mod_behavior_ai:
                now = time.time()
                cooldown_ok = (now - st.session_state["last_vlm_call"]) >= 8
                motion_spike = motion_score > 0.02
                periodic_due = (now - last_general_scan) >= 20
                if cooldown_ok and (motion_spike or periodic_due):
                    st.session_state["last_vlm_call"] = now
                    last_general_scan = now
                    is_abnormal, desc = scan_for_abnormal_behavior(frame)
                    if is_abnormal:
                        highlight_zone(frame,(int(W*0.05),HUD,int(W*0.95),H-1),"ABNORMAL BEHAVIOR DETECTED","HIGH")
                        s=snap(frame); add_event("BEHAVIOR","Abnormal Behavior Detected","HIGH",cam_label,desc or "Unusual activity flagged by AI scan",s)

            # ── Fighting heuristic ──────────────────────────────────────────
            # Pre-filter: 2+ persons within 150px of each other AND motion spike
            if mod_behavior_ai and len(pcents) >= 2 and motion_score > 0.01:
                _fight_close = any(
                    ((pcents[i][0]-pcents[j][0])**2+(pcents[i][1]-pcents[j][1])**2)**0.5 < 150
                    for i in range(len(pcents)) for j in range(i+1, len(pcents))
                )
                if _fight_close:
                    now_f = time.time()
                    if (now_f - st.session_state["last_vlm_call"]) >= 8 and (now_f - last_fight_scan) >= 8:
                        st.session_state["last_vlm_call"] = now_f; last_fight_scan = now_f
                        is_f, desc = scan_for_fight(frame)
                        if is_f:
                            highlight_zone(frame,(int(W*0.05),HUD,int(W*0.95),H-1),"PHYSICAL ALTERCATION DETECTED","HIGH")
                            s=snap(frame); add_event("BEHAVIOR","Fighting / Physical Altercation","HIGH",cam_label,desc or "Persons in close contact with high motion",s)

            # ── Pickpocket / bag-snatch heuristic ───────────────────────────
            # Pre-filter: 2+ persons within 150px AND moderate motion
            if mod_behavior_ai and len(pcents) >= 2 and motion_score > 0.01:
                _pick_close = any(
                    ((pcents[i][0]-pcents[j][0])**2+(pcents[i][1]-pcents[j][1])**2)**0.5 < 150
                    for i in range(len(pcents)) for j in range(i+1, len(pcents))
                )
                if _pick_close:
                    now_p = time.time()
                    if (now_p - st.session_state["last_vlm_call"]) >= 8 and (now_p - last_pick_scan) >= 8:
                        st.session_state["last_vlm_call"] = now_p; last_pick_scan = now_p
                        is_pk, desc = scan_for_pickpocket(frame)
                        if is_pk:
                            highlight_zone(frame,(int(W*0.05),HUD,int(W*0.95),H-1),"POSSIBLE THEFT — REVIEW ADVISED","HIGH")
                            s=snap(frame); add_event("BEHAVIOR","Possible Pickpocketing / Bag-snatch — Needs Review","HIGH",cam_label,desc or "Close contact with suspected item movement",s)

            # ── Fallen person heuristic ──────────────────────────────────────
            # Pre-filter: any person bounding box is wider than tall (horizontal = collapsed)
            if mod_behavior_ai:
                _fallen = []
                if result is not None and result.boxes is not None:
                    for _b in result.boxes:
                        if int(_b.cls[0]) != PERSON_CLS: continue
                        _bx1,_by1,_bx2,_by2 = map(int, _b.xyxy[0].tolist())
                        _bw,_bh = _bx2-_bx1, _by2-_by1
                        if _bw > _bh * 1.4 and _bw > 60 and _bh > 15:
                            _fallen.append((_bx1,_by1,_bx2,_by2))
                if _fallen:
                    now_fl = time.time()
                    if (now_fl - st.session_state["last_vlm_call"]) >= 8 and (now_fl - last_fall_scan) >= 8:
                        st.session_state["last_vlm_call"] = now_fl; last_fall_scan = now_fl
                        _fb = _fallen[0]
                        _crop = frame[max(0,_fb[1]-10):_fb[3]+10, max(0,_fb[0]-10):_fb[2]+10]
                        if _crop.size == 0: _crop = frame
                        is_fl2, desc = scan_for_fallen_person(_crop)
                        if is_fl2:
                            highlight_zone(frame,(_fb[0],_fb[1],_fb[2],_fb[3]),"PERSON FALLEN — ASSISTANCE NEEDED","HIGH")
                            s=snap(frame); add_event("BEHAVIOR","Person Fallen / Collapsed","HIGH",cam_label,desc or "Person appears to have fallen",s)

            # ── Suspicious package drop heuristic ───────────────────────────
            # Pre-filter: object appears for the first time near a previous person position,
            # and that person has now walked >200px away.
            if mod_behavior_ai and has_ids and result is not None and result.boxes is not None:
                _live_tids = set()
                for _b in result.boxes:
                    _cid2 = int(_b.cls[0])
                    if _cid2 == PERSON_CLS: continue
                    if model.names.get(_cid2, "") in THREATS: continue
                    if _b.id is None or len(_b.id) == 0: continue
                    _tid2 = int(_b.id[0]); _live_tids.add(_tid2)
                    _bx1,_by1,_bx2,_by2 = map(int, _b.xyxy[0].tolist())
                    _ocx,_ocy = (_bx1+_bx2)//2, (_by1+_by2)//2
                    if _tid2 not in pkg_close_on_drop:
                        _near_prev = min((((_ocx-px)**2+(_ocy-py)**2)**0.5 for px,py in prev_pcents), default=9999)
                        pkg_close_on_drop[_tid2] = _near_prev < 150
                    if pkg_close_on_drop.get(_tid2, False):
                        _near_now = min((((_ocx-px)**2+(_ocy-py)**2)**0.5 for px,py in pcents), default=9999)
                        if _near_now > 200:
                            now_dr = time.time()
                            if (now_dr - st.session_state["last_vlm_call"]) >= 8 and (now_dr - last_drop_scan) >= 8:
                                st.session_state["last_vlm_call"] = now_dr; last_drop_scan = now_dr
                                _crop = frame[max(0,_by1-20):_by2+20, max(0,_bx1-20):_bx2+20]
                                if _crop.size == 0: _crop = frame
                                _lbl2 = model.names.get(_cid2, "object")
                                is_sd, desc = scan_for_package_drop(_crop)
                                if is_sd:
                                    highlight_zone(frame,(_bx1,_by1,_bx2,_by2),"SUSPICIOUS PACKAGE DROP","HIGH")
                                    s=snap(frame); add_event("BEHAVIOR","Suspicious Package Drop","HIGH",cam_label,desc or f"'{_lbl2}' dropped, person walked away",s)
                                pkg_close_on_drop[_tid2] = False  # fire only once per object
                pkg_close_on_drop = {k: v for k, v in pkg_close_on_drop.items() if k in _live_tids}

            # ── Unattended child heuristic ────────────────────────────────────
            # Pre-filter: at least 1 person in frame. Scans every 30s (still consumes shared cooldown).
            if mod_behavior_ai and person_count >= 1:
                now_ch = time.time()
                if (now_ch - st.session_state["last_vlm_call"]) >= 8 and (now_ch - last_child_scan) >= 30:
                    st.session_state["last_vlm_call"] = now_ch; last_child_scan = now_ch
                    is_ch, desc = scan_for_unattended_child(frame)
                    if is_ch:
                        highlight_zone(frame,(int(W*0.05),HUD,int(W*0.95),H-1),"UNATTENDED CHILD DETECTED","MEDIUM")
                        s=snap(frame); add_event("BEHAVIOR","Unattended Child","MEDIUM",cam_label,desc or "Child appears to be alone and unattended",s)

            prev_pcents = list(pcents)  # snapshot for next frame's package-drop proximity check

            # Triggers
            if mod_restricted:
                rz=sum(1 for cx,cy in pcents if pt_in(cx,cy,restr_zone))
                if rz:
                    highlight_zone(frame,restr_zone,"RESTRICTED ZONE BREACH","HIGH")
                    s=snap(frame); add_event("SECURITY","Restricted Zone Intrusion","HIGH",cam_label,f"{rz} person(s)",s)

            if mod_track:
                tz=sum(1 for cx,cy in pcents if pt_in(cx,cy,track_zone))
                if tz:
                    highlight_zone(frame,track_zone,"TRACK INTRUSION","CRITICAL")
                    s=snap(frame); add_event("SECURITY","Track Intrusion","CRITICAL",cam_label,f"{tz} person(s) near track edge",s)

            if mod_staff:
                staff_empty=staff_empty+1 if staff_count==0 else 0
                if staff_empty>int(src_fps*7):
                    highlight_zone(frame,staff_zone,"STAFF POST VACANT","MEDIUM")
                    s=snap(frame); add_event("OPERATIONS","Staff Post Unattended","MEDIUM",cam_label,"Post vacant >7s",s)

            if zone_count>=crowd_limit:
                risk,rc_cv="CRITICAL",(150,0,0)
                highlight_zone(frame,crowd_zone,"OVERCROWDING","CRITICAL")
                s=snap(frame); add_event("CROWD","Overcrowding","HIGH",cam_label,f"{zone_count} in zone (limit {crowd_limit})",s)
            elif zone_count>=max(1, min(crowd_limit - 1, int(crowd_limit * 0.70))):
                risk,rc_cv="WARNING",(20,100,185)
                highlight_zone(frame,crowd_zone,"CROWD WARNING","MEDIUM")
            else:
                risk,rc_cv="NORMAL",(0,120,40)

            # Minimal HUD
            cv2.rectangle(frame,(0,0),(W,HUD),(9,9,9),-1)
            cv2.line(frame,(0,HUD),(W,HUD),(28,28,28),1)
            # Left
            cv2.putText(frame,"RAILVISION",(10,18),cv2.FONT_HERSHEY_SIMPLEX,0.40,(60,60,60),1,cv2.LINE_AA)
            cv2.putText(frame,cam_label,(10,33),cv2.FONT_HERSHEY_SIMPLEX,0.32,(50,50,50),1,cv2.LINE_AA)
            cv2.putText(frame,_ts(),(10,48),cv2.FONT_HERSHEY_SIMPLEX,0.30,(40,40,40),1,cv2.LINE_AA)
            # Centre: person count
            lp=f"PERSONS DETECTED: {person_count}"
            (tw,_),_=cv2.getTextSize(lp,cv2.FONT_HERSHEY_SIMPLEX,0.80,2)
            cv2.putText(frame,lp,((W-tw)//2,50),cv2.FONT_HERSHEY_SIMPLEX,0.80,(210,210,210),2,cv2.LINE_AA)
            # Right: zone + status badge
            lz=f"ZONE: {zone_count}"
            (zw,_),_=cv2.getTextSize(lz,cv2.FONT_HERSHEY_SIMPLEX,0.46,1)
            cv2.putText(frame,lz,(W-zw-10,26),cv2.FONT_HERSHEY_SIMPLEX,0.46,(80,140,140),1,cv2.LINE_AA)
            bw=88; cv2.rectangle(frame,(W-bw-8,34),(W-8,HUD-4),rc_cv,-1)
            (brw,_),_=cv2.getTextSize(risk,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
            cv2.putText(frame,risk,(W-bw//2-brw//2-8,HUD-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1,cv2.LINE_AA)
            if risk=="CRITICAL":
                t2=frame.copy(); cv2.rectangle(t2,(0,0),(W,H),(100,0,0),-1)
                frame[:]=cv2.addWeighted(t2,0.04,frame,0.96,0)
                msg="CRITICAL — OVERCROWDING — IMMEDIATE RESPONSE REQUIRED"
                (mw,_),_=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.46,1)
                cv2.putText(frame,msg,((W-mw)//2,H-14),cv2.FONT_HERSHEY_SIMPLEX,0.46,(180,30,30),1,cv2.LINE_AA)

            frame_slot.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),channels="RGB",width="stretch")

            now_t=time.time(); dt_=now_t-t_prev
            proc_fps=(1.0/dt_) if dt_>0 else proc_fps; t_prev=now_t
            stat_slot.markdown(statstrip_html(person_count,zone_count,risk,
                               st.session_state["total_alerts"],proc_fps,faces_blurred),
                               unsafe_allow_html=True)

            rl = {"CRITICAL":"r","WARNING":"a","NORMAL":"g"}.get(risk,"m")
            sp_cls="g" if staff_count>0 else ("r" if staff_empty>int(src_fps*7) else "m")
            sp_val="OCCUPIED" if staff_count>0 else ("VACANT" if staff_empty>int(src_fps*7) else "EMPTY")
            th_v,th_c = ("⚠ DETECTED","r") if threat_hits else ("CLEAR","g")
            status_slot.markdown(status_rows([
                ("Threat Status",    th_v,            th_c),
                ("Crowd Status",     risk,           rl),
                ("Zone Count",       str(zone_count), ""),
                ("Total Persons",    str(person_count),""),
                ("Staff Post",       sp_val,          sp_cls),
                ("Faces Blurred",    str(faces_blurred),""),
                ("AI Engine",        "● ACTIVE",      "g"),
            ]), unsafe_allow_html=True)

            evts=st.session_state["events"]
            incident_slot.markdown(f"<div class='rv-log-scroll'>{alerts_html(evts, 8)}</div>", unsafe_allow_html=True)

            st.session_state["live"]={"people":person_count,"zone":zone_count,"risk":risk}
            hist=st.session_state["history"]
            hist.append({"t":_ts(),"people":person_count,"zone":zone_count})
            st.session_state["history"]=hist[-150:]

            if prog and total_fr>0: prog.progress(min(frame_idx/total_fr,1.0))

        finally:
            cap.release()
            try:
                if prog: prog.empty()
            except Exception: pass
            if tmp_path:
                try: Path(tmp_path).unlink(missing_ok=True)
                except Exception: pass
        
        try:
            stat_slot.markdown(statstrip_html(person_count,zone_count,"COMPLETE",
                               st.session_state["total_alerts"],0,faces_total), unsafe_allow_html=True)
            if not st.session_state.get("stop_requested", False):
                st.success(f"Complete — {frame_idx} frames analysed.")
        except Exception:
            pass
