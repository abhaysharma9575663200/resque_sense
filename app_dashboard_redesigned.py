"""
RESQ_SENSE Mission Control Dashboard.
Streamlit UI for person detection, media analysis, and incident reporting.
Posture-based priority indicators are experimental visual heuristics, not medical triage.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from resq_sense.config import (
    DEFAULT_MODEL_NAME, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD,
    AVAILABLE_FILTERS, MODELS_DIR, SNAPSHOTS_DIR, INCIDENT_LOG_CSV,
)
from resq_sense.engine import ResQSenseEngine
from resq_sense.vision_filters import apply_filter
from resq_sense.tracker import SurvivorTracker
from resq_sense.incident_logger import IncidentLogger


st.set_page_config(
    page_title="RESQ_SENSE | Mission Control",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_dashboard_theme(theme: str) -> None:
    if theme == "Dark":
        colors = {
            "bg": "#071B35", "panel": "#102A49", "panel_alt": "#163657",
            "text": "#F4F8FC", "muted": "#A9BCD2", "border": "#294969",
            "blue": "#4D8CCB", "red": "#E53945", "input": "#102A49",
        }
    else:
        colors = {
            "bg": "#F2F6FB", "panel": "#FFFFFF", "panel_alt": "#EAF2FA",
            "text": "#102A49", "muted": "#60758C", "border": "#D5E0EC",
            "blue": "#315F91", "red": "#E53945", "input": "#FFFFFF",
        }

    css = """
    <style>
    :root {
      --bg: __BG__; --panel: __PANEL__; --panel-alt: __PANEL_ALT__;
      --text: __TEXT__; --muted: __MUTED__; --border: __BORDER__;
      --blue: __BLUE__; --red: __RED__; --input: __INPUT__;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
      background: var(--bg); color: var(--text);
      font-family: Inter, "Segoe UI", Arial, sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1800px; padding-top: 1rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
      background: var(--panel); border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    h1, h2, h3, h4, p, label, span { color: var(--text); }
    [data-testid="stCaptionContainer"] p { color: var(--muted); }

    .command-topbar {
      display:flex; align-items:center; justify-content:space-between;
      flex-wrap:wrap; gap:18px; width:100%; min-height:70px;
      padding:11px 18px; margin-bottom:16px; background:var(--panel);
      border:1px solid var(--border); border-top:2px solid var(--blue);
      border-radius:7px;
    }
    .command-brand { display:flex; align-items:center; gap:11px; min-width:220px; }
    .brand-mark {
      display:flex; align-items:center; justify-content:center; width:38px;
      height:38px; background:var(--blue); color:white; font-size:20px;
      font-weight:800; border-radius:5px;
    }
    .brand-name { color:var(--text); font-size:1rem; font-weight:800; letter-spacing:.08em; }
    .brand-subtitle { color:var(--muted); font-size:.61rem; letter-spacing:.09em; margin-top:3px; }
    .command-context { display:flex; flex-direction:column; gap:4px; }
    .context-label { color:var(--muted); font-size:.60rem; letter-spacing:.12em; }
    .context-value { color:var(--text); font-size:.76rem; font-weight:700; letter-spacing:.04em; }
    .command-interface { color:var(--muted); font-size:.68rem; font-weight:700; letter-spacing:.08em; white-space:nowrap; }
    .interface-indicator { display:inline-block; width:7px; height:7px; margin-right:6px; background:var(--blue); border-radius:50%; }

    .section-kicker { color:var(--blue); font-size:.68rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; margin-bottom:4px; }
    .workspace-title { font-size:1.42rem; font-weight:750; margin:0 0 2px 0; }
    .workspace-subtitle { color:var(--muted); font-size:.82rem; margin-bottom:14px; }
    .panel { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:13px 15px; margin-bottom:12px; }
    .panel-heading { color:var(--muted); font-size:.67rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase; margin-bottom:9px; }
    .status-line { display:flex; align-items:center; gap:8px; font-weight:700; }
    .status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--blue); }
    .big-number { font-size:1.7rem; font-weight:800; line-height:1.2; color:var(--text); }
    .priority-critical { border-left:3px solid var(--red); background:#E5394518; padding:9px 11px; border-radius:5px; margin:6px 0; }
    .priority-warning { border-left:3px solid var(--blue); background:#4D8CCB18; padding:9px 11px; border-radius:5px; margin:6px 0; }
    .priority-normal { border-left:3px solid var(--border); background:var(--panel-alt); padding:9px 11px; border-radius:5px; margin:6px 0; }
    .evidence-note { color:var(--muted); font-size:.72rem; line-height:1.45; border-top:1px solid var(--border); padding-top:9px; margin-top:10px; }

    [data-testid="stMetric"] { background:var(--panel); border:1px solid var(--border); border-left:3px solid var(--blue); border-radius:7px; padding:11px 13px; }
    [data-testid="stMetricLabel"] { color:var(--muted); font-size:.70rem; text-transform:uppercase; letter-spacing:.07em; }
    [data-testid="stMetricValue"] { color:var(--text); font-weight:750; }
    [data-testid="stTabs"] [role="tablist"] { gap:7px; border-bottom:1px solid var(--border); }
    [data-testid="stTabs"] button[role="tab"] { color:var(--muted); font-weight:650; }
    [data-testid="stTabs"] button[aria-selected="true"] { color:var(--blue); border-bottom:2px solid var(--red); }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"], [data-testid="stTextArea"] textarea {
      background:var(--input); color:var(--text); border-color:var(--border); border-radius:6px;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid var(--border); border-radius:7px; overflow:hidden; }
    [data-testid="stFileUploader"] { background:var(--panel); border:1px dashed var(--border); border-radius:7px; padding:7px; }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button { border-radius:6px; font-weight:650; border:1px solid var(--border); }
    hr { border-color:var(--border); }
    @media (max-width:900px) { .command-topbar { align-items:flex-start; } .command-context { min-width:42%; } }
    </style>
    """
    replacements = {
        "__BG__": colors["bg"], "__PANEL__": colors["panel"],
        "__PANEL_ALT__": colors["panel_alt"], "__TEXT__": colors["text"],
        "__MUTED__": colors["muted"], "__BORDER__": colors["border"],
        "__BLUE__": colors["blue"], "__RED__": colors["red"],
        "__INPUT__": colors["input"],
    }
    for key, value in replacements.items():
        css = css.replace(key, value)
    st.markdown(css, unsafe_allow_html=True)


@st.cache_resource
def get_engine(model_path: str):
    return ResQSenseEngine(model_path=model_path)


def render_topbar():
    st.markdown(
        """
        <div class="command-topbar">
          <div class="command-brand"><div class="brand-mark">R</div><div>
            <div class="brand-name">RESQ_SENSE</div>
            <div class="brand-subtitle">DISASTER INTELLIGENCE PLATFORM</div>
          </div></div>
          <div class="command-context"><span class="context-label">OPERATION</span>
            <span class="context-value">SEARCH &amp; RESCUE</span></div>
          <div class="command-context"><span class="context-label">WORKSPACE</span>
            <span class="context-value">HUMAN DETECTION</span></div>
          <div class="command-interface"><span class="interface-indicator"></span>
            LOCAL CONTROL INTERFACE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.markdown("## RESQ_SENSE")
    st.sidebar.caption("FIELD OPERATIONS / CONTROL PANEL")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 01 / MISSION & CAMERA")
    source_type = st.sidebar.radio(
        "Video Feed Source",
        ["Live Webcam (Native OpenCV)", "Browser Camera Snapshot"],
        index=0,
        help="Choose continuous local webcam processing or browser snapshots.",
    )
    if source_type == "Live Webcam (Native OpenCV)":
        camera_index = st.sidebar.selectbox(
            "Camera Device", [0, 1, 2],
            format_func=lambda x: f"Camera {x} (Built-in Webcam)" if x == 0 else f"Camera {x} (External USB Camera)",
            index=0,
        )
        run_stream = st.sidebar.checkbox("START LIVE STREAM", value=False)
    else:
        camera_index, run_stream = 0, False

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 02 / AI MODEL")
    models = [DEFAULT_MODEL_NAME]
    if MODELS_DIR.exists():
        for model_file in MODELS_DIR.glob("*.pt"):
            if model_file.name not in models:
                models.append(model_file.name)
    model_choice = st.sidebar.selectbox("Detection Model", models, index=0)
    uploaded_model = st.sidebar.file_uploader("Upload Custom Weights", type=["pt"])
    active_model = model_choice
    if uploaded_model is not None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        custom_path = MODELS_DIR / uploaded_model.name
        with open(custom_path, "wb") as file:
            file.write(uploaded_model.getbuffer())
        st.sidebar.success(f"Custom model saved: {uploaded_model.name}")
        active_model = str(custom_path)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 03 / IMAGE ENHANCEMENT")
    selected_filter = st.sidebar.selectbox("Preprocessing Mode", AVAILABLE_FILTERS, index=0)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 04 / DETECTION PARAMETERS")
    confidence = st.sidebar.slider("Confidence Threshold", 0.10, 0.95, value=DEFAULT_CONF_THRESHOLD, step=0.05)
    iou = st.sidebar.slider("IoU Threshold (NMS)", 0.20, 0.80, value=DEFAULT_IOU_THRESHOLD, step=0.05)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 05 / SECTOR & ALERTS")
    sector = st.sidebar.text_input("Active Sector", "Sector-Alpha-1")
    audio_enabled = st.sidebar.checkbox("Enable Audio Alert", value=True)
    st.sidebar.markdown("---")
    st.sidebar.caption("RESQ_SENSE / MISSION CONTROL")
    return source_type, camera_index, run_stream, active_model, selected_filter, confidence, iou, sector, audio_enabled


def count_priorities(items):
    high = sum(1 for item in items if item.get("priority_rank") == 1)
    review = sum(1 for item in items if item.get("priority_rank") == 2)
    other = sum(1 for item in items if item.get("priority_rank") == 3)
    return high, review, other


def render_intelligence_panel(target, detections, latency=None, fps=None, state="STANDBY"):
    with target:
        st.markdown('<div class="panel-heading">LIVE INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-line"><span class="status-dot"></span>{state}</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.caption("DETECTED PERSONS")
        st.markdown(f'<div class="big-number">{len(detections)}</div>', unsafe_allow_html=True)
        if latency is not None:
            st.caption(f"INFERENCE LATENCY · {latency:.1f} ms")
        if fps is not None:
            st.caption(f"STREAM RATE · {fps:.1f} FPS")
        high, review, other = count_priorities(detections)
        st.markdown("#### Visual priority indicators")
        if high:
            st.markdown(f'<div class="priority-critical"><b>HIGH PRIORITY</b><br>{high} posture-based flag(s)</div>', unsafe_allow_html=True)
        if review:
            st.markdown(f'<div class="priority-warning"><b>REVIEW</b><br>{review} posture-based flag(s)</div>', unsafe_allow_html=True)
        if other:
            st.markdown(f'<div class="priority-normal"><b>OTHER DETECTIONS</b><br>{other} posture-based flag(s)</div>', unsafe_allow_html=True)
        if not detections:
            st.info("No person detections in the current frame.")
        st.markdown(
            '<div class="evidence-note">Experimental visual heuristics based on detection geometry. Not medical triage, proof of entrapment, or a substitute for responder assessment.</div>',
            unsafe_allow_html=True,
        )


def main():
    theme = st.sidebar.radio("Interface Theme", ["Dark", "Light"], horizontal=True, key="dashboard_theme")
    apply_dashboard_theme(theme)
    render_topbar()

    (
        source_type, camera_index, run_stream, active_model, selected_filter,
        confidence, iou, sector, audio_enabled,
    ) = render_sidebar()

    try:
        engine = get_engine(active_model)
    except Exception as error:
        st.error(f"Failed to load AI model '{active_model}': {error}")
        st.stop()

    tracker = SurvivorTracker()
    logger = IncidentLogger()

    tab_live, tab_media, tab_logs, tab_research = st.tabs([
        "◉ LIVE OPERATIONS", "▧ IMAGE & FOOTAGE", "▤ INCIDENT LOG", "⌘ RESEARCH FRAMEWORK",
    ])

    # TAB 1 — LIVE OPERATIONS
    with tab_live:
        st.markdown('<div class="section-kicker">01 / Operational workspace</div>', unsafe_allow_html=True)
        st.markdown('<div class="workspace-title">Human Detection Operations</div>', unsafe_allow_html=True)
        st.markdown('<div class="workspace-subtitle">Local camera analysis · Person detection · Visual evidence logging</div>', unsafe_allow_html=True)

        metric_cols = st.columns(4)
        fps_metric, latency_metric, people_metric, priority_metric = [col.empty() for col in metric_cols]
        fps_metric.metric("Stream FPS", "--")
        latency_metric.metric("Inference latency", "-- ms")
        people_metric.metric("Detected persons", "0")
        priority_metric.metric("Visual priority", "STANDBY")

        camera_col, intel_col = st.columns([3.25, 1.05], gap="medium")
        with camera_col:
            st.markdown('<div class="panel-heading">PRIMARY SENSOR / CAMERA FEED</div>', unsafe_allow_html=True)
            feed = st.empty()
        intel = intel_col.empty()

        if source_type == "Live Webcam (Native OpenCV)":
            if not run_stream:
                feed.info("STREAM STANDBY\n\nSelect START LIVE STREAM in the left control panel to begin local camera detection.")
                render_intelligence_panel(intel, [], state="STANDBY")
            else:
                capture = cv2.VideoCapture(int(camera_index))
                if not capture.isOpened():
                    capture = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
                if not capture.isOpened():
                    feed.error(f"Cannot access Camera {camera_index}. Check the selected device and close other applications using the camera.")
                    render_intelligence_panel(intel, [], state="CAMERA UNAVAILABLE")
                else:
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    previous_time = time.time()
                    fps_value = 0.0
                    try:
                        while run_stream:
                            ok, frame = capture.read()
                            if not ok or frame is None:
                                feed.warning("Frame read failed or camera disconnected.")
                                break
                            now = time.time()
                            fps_value = 0.9 * fps_value + 0.1 / max(1e-5, now - previous_time)
                            previous_time = now

                            enhanced = apply_filter(frame, selected_filter)
                            detections, latency_ms = engine.detect(
                                enhanced, conf_threshold=confidence, iou_threshold=iou,
                            )
                            tracked = tracker.update(detections)
                            for detection in tracked:
                                logger.log_detection(frame, detection, grid_sector=sector)

                            # fps is shown in the metric; draw_hud receives only supported display arguments.
                            hud = engine.draw_hud(
                                enhanced, tracked, latency_ms=latency_ms,
                                filter_name=selected_filter, sector_name=sector,
                            )
                            feed.image(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                            high, review, other = count_priorities(tracked)
                            fps_metric.metric("Stream FPS", f"{fps_value:.1f}")
                            latency_metric.metric("Inference latency", f"{latency_ms:.1f} ms")
                            people_metric.metric("Detected persons", str(len(tracked)))
                            priority_metric.metric("Visual priority", "HIGH" if high else "REVIEW" if review else "OTHER" if other else "NONE")
                            render_intelligence_panel(intel, tracked, latency_ms, fps_value, "PROCESSING")
                            time.sleep(0.01)
                    finally:
                        capture.release()
        else:
            snapshot = st.camera_input("Capture a browser camera snapshot")
            if snapshot is None:
                feed.info("CAMERA READY\n\nCapture a snapshot to run person detection.")
                render_intelligence_panel(intel, [], state="AWAITING SNAPSHOT")
            else:
                raw = np.frombuffer(snapshot.getvalue(), np.uint8)
                frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if frame is not None:
                    enhanced = apply_filter(frame, selected_filter)
                    detections, latency_ms = engine.detect(enhanced, conf_threshold=confidence, iou_threshold=iou)
                    tracked = tracker.update(detections)
                    for detection in tracked:
                        logger.log_detection(frame, detection, grid_sector=sector)
                    hud = engine.draw_hud(enhanced, tracked, latency_ms=latency_ms, filter_name=selected_filter, sector_name=sector)
                    feed.image(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    latency_metric.metric("Inference latency", f"{latency_ms:.1f} ms")
                    people_metric.metric("Detected persons", str(len(tracked)))
                    high, review, other = count_priorities(tracked)
                    priority_metric.metric("Visual priority", "HIGH" if high else "REVIEW" if review else "OTHER" if other else "NONE")
                    render_intelligence_panel(intel, tracked, latency_ms, state="SNAPSHOT ANALYSED")

        st.markdown("---")
        left_note, right_note = st.columns([2, 1])
        left_note.caption(f"ACTIVE SECTOR: {sector} · PREPROCESSING: {selected_filter}")
        right_note.caption("Model output requires field validation.")

    # TAB 2 — IMAGE & FOOTAGE
    with tab_media:
        st.markdown('<div class="section-kicker">02 / Evidence review</div>', unsafe_allow_html=True)
        st.markdown('<div class="workspace-title">Image & Footage Analysis</div>', unsafe_allow_html=True)
        st.caption("Upload reconnaissance photos, drone snapshots, or rubble images for person detection.")
        uploaded = st.file_uploader("Upload image evidence", type=["jpg", "jpeg", "png", "bmp"], key="media_image_upload")
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            original_col, result_col = st.columns(2, gap="medium")
            with original_col:
                st.markdown("#### SOURCE IMAGE")
                st.image(image, use_container_width=True)

            enhanced = apply_filter(frame, selected_filter)
            detections, latency_ms = engine.detect(enhanced, conf_threshold=confidence, iou_threshold=iou)
            tracked = tracker.update(detections)
            for detection in tracked:
                logger.log_detection(frame, detection, grid_sector=sector)
            hud = engine.draw_hud(enhanced, tracked, latency_ms=latency_ms, filter_name=selected_filter, sector_name=sector)
            with result_col:
                st.markdown("#### MODEL OUTPUT")
                st.image(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            a, b, c = st.columns(3)
            a.metric("Persons detected", len(tracked))
            b.metric("Inference latency", f"{latency_ms:.1f} ms")
            c.metric("Active sector", sector)
            st.caption("Posture/priority labels are experimental visual heuristics, not medical triage.")
            if tracked:
                rows = []
                for detection in tracked:
                    rows.append({
                        "Tracker ID": f"ID #{detection.get('tracker_id', 'N/A')}",
                        "Visual Priority": detection.get("triage_level", "N/A"),
                        "Posture": detection.get("posture", "N/A"),
                        "Confidence": f"{detection.get('confidence', 0) * 100:.1f}%",
                        "Aspect Ratio (W/H)": detection.get("aspect_ratio", "N/A"),
                        "Bounding Box": str(detection.get("box", "N/A")),
                    })
                st.markdown("#### DETECTION RECORDS")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # TAB 3 — INCIDENT LOG
    with tab_logs:
        st.markdown('<div class="section-kicker">03 / Mission records</div>', unsafe_allow_html=True)
        st.markdown('<div class="workspace-title">Incident Log & Evidence</div>', unsafe_allow_html=True)
        st.caption("Recorded detections, timestamps, sector labels, and saved image evidence.")
        refresh_col, export_col = st.columns([1, 4])
        with refresh_col:
            if st.button("↻ Refresh records"):
                st.rerun()
        if INCIDENT_LOG_CSV.exists():
            dataframe = pd.read_csv(INCIDENT_LOG_CSV)
            if not dataframe.empty:
                st.dataframe(dataframe.tail(100), use_container_width=True)
                csv_data = dataframe.to_csv(index=False).encode("utf-8")
                with export_col:
                    st.download_button(
                        "Download mission log (CSV)", data=csv_data,
                        file_name=f"resq_sense_mission_log_{int(time.time())}.csv",
                        mime="text/csv",
                    )
            else:
                st.info("The incident log exists but contains no records yet.")
        else:
            st.info("No incident log has been created yet.")
        st.markdown("#### RECENT SNAPSHOT EVIDENCE")
        recent = list(SNAPSHOTS_DIR.glob("*.jpg")) if SNAPSHOTS_DIR.exists() else []
        recent.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        if recent:
            gallery = st.columns(4)
            for index, snapshot_path in enumerate(recent[:8]):
                with gallery[index % 4]:
                    st.image(str(snapshot_path), caption=snapshot_path.stem[:28], use_container_width=True)
        else:
            st.caption("No snapshots captured yet.")

    # TAB 4 — RESEARCH FRAMEWORK
    with tab_research:
        st.markdown('<div class="section-kicker">04 / Technical basis</div>', unsafe_allow_html=True)
        st.markdown('<div class="workspace-title">Research Framework</div>', unsafe_allow_html=True)
        st.markdown(
            """
            This software is presented in relation to the framework outlined in:

            > **"Human Body Detection in Disaster Environments: A Review of Framework Design and Development Strategies"**  
            > *Vivek Sharma, Dr. Manish Dixit (September 2025)*  
            > *Journal of Global Research in Electronics and Communication, 1(9), 01-06.*

            ### Addressed design challenges

            **1. Variable body postures**
            - Bounding-box geometry is used to derive experimental posture/priority indicators.
            - These indicators do not establish injury severity, entrapment, or medical condition.

            **2. Adverse environmental visibility**
            - Available preprocessing filters can be applied before model inference.
            - Filter performance must be evaluated on representative smoke, dust, darkness, and rubble imagery.

            **3. Real-time processing**
            - The application measures inference latency and estimates stream FPS during webcam processing.
            - Actual performance depends on hardware, model weights, image size, and scene conditions.

            **4. Evidence and reporting**
            - Detection records and snapshot crops can support review.
            - Model outputs require human verification before operational decisions.
            """
        )


if __name__ == "__main__":
    main()
