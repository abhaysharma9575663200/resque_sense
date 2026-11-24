"""
ResQ_sense: Mission Control Web Dashboard (Landing-page style UI).
Interactive Streamlit application for disaster human detection, live webcam execution,
custom model management, triage analysis, and incident reporting.

Based on: Sharma & Dixit (2025), "Human Body Detection in Disaster Environments"
"""

import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from resq_sense.config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    AVAILABLE_FILTERS,
    MODELS_DIR,
    SNAPSHOTS_DIR,
    INCIDENT_LOG_CSV,
)
from resq_sense.engine import ResQSenseEngine
from resq_sense.vision_filters import apply_filter
from resq_sense.tracker import SurvivorTracker
from resq_sense.incident_logger import IncidentLogger

st.set_page_config(
    page_title="Human Detection in Disaster",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# THEME: black page, purple nav, serif hero, purple glow bottom-left
# ----------------------------------------------------------------------------
PURPLE = "#8f2cff"
RED = "#e53935"

st.markdown(
    f"""
    <style>
    /* ---------- Page background ---------- */
    .stApp {{
        background:
            radial-gradient(circle at 4% 78%, rgba(100, 30, 160, 0.55) 0%, rgba(60, 15, 110, 0.25) 22%, transparent 45%),
            #000000;
        color: #e6e6ea;
        font-family: "Segoe UI", Arial, Helvetica, sans-serif;
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stToolbar"], #MainMenu, footer {{ display: none !important; visibility: hidden; }}
    .block-container {{ padding-top: 1.2rem; max-width: 1500px; }}

    /* ---------- Site title ---------- */
    .site-title {{
        text-align: center;
        margin: 0.2rem 0 1.4rem 0;
    }}
    .site-title h1 {{
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.2px;
    }}
    .site-title p {{
        margin: 0.25rem 0 0 0;
        font-size: 1.45rem;
        color: #ffffff;
    }}

    /* ---------- Top navigation (tabs) ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2.6rem;
        padding: 0.3rem 1.2rem;
        border-bottom: none;
        align-items: center;
    }}
    .stTabs [data-baseweb="tab-border"] {{ display: none; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: {PURPLE}; height: 3px; }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        padding: 0.5rem 0;
        height: auto;
    }}
    .stTabs [data-baseweb="tab"] p {{
        color: {PURPLE};
        font-size: 1.35rem;
        font-weight: 700;
    }}
    .stTabs [data-baseweb="tab"]:hover p {{ color: #b57bff; }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] p {{ color: #b57bff; }}

    /* Last tab = red "Mission" pill pushed to the right */
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"]:last-of-type {{
        margin-left: auto;
        background: {RED};
        border-radius: 6px;
        padding: 0.65rem 1.1rem;
    }}
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"]:last-of-type p {{
        color: #ffffff;
    }}
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"]:last-of-type:hover {{
        background: #f44a46;
    }}

    /* ---------- Hero ---------- */
    .hero {{
        text-align: center;
        padding: 3.2rem 1rem 0.5rem 1rem;
    }}
    .hero h1 {{
        font-family: Georgia, "Times New Roman", serif;
        font-weight: 700;
        font-size: clamp(2.4rem, 6.2vw, 5.4rem);
        line-height: 1.18;
        color: #b9b9bf;
        margin: 0;
    }}
    .hero .tagline {{
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.7rem;
        color: #c9c9cf;
        margin: 2.2rem 0 3.2rem 0;
    }}
    .feature-card {{
        max-width: 1020px;
        margin: 0 auto 2rem auto;
        padding: 2.2rem 1.5rem 1.4rem 1.5rem;
        border-radius: 26px;
        border: 1px solid rgba(143, 44, 255, 0.18);
        background: rgba(12, 0, 24, 0.65);
        text-align: center;
    }}
    .feature-card p {{
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.7rem;
        color: #d2d2d8;
        margin: 0 0 1.6rem 0;
    }}

    /* ---------- Home sections ---------- */
    .section-title {{
        font-family: Georgia, "Times New Roman", serif;
        font-size: 2.3rem; color: #d9d9df; text-align: center;
        margin: 3.2rem 0 1.6rem 0;
    }}
    .grid {{
        display: grid; gap: 1.2rem; max-width: 1150px; margin: 0 auto;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }}
    .card {{
        background: rgba(14, 4, 28, 0.75);
        border: 1px solid rgba(143, 44, 255, 0.22);
        border-radius: 16px; padding: 1.4rem 1.3rem;
    }}
    .card h4 {{ margin: 0 0 0.5rem 0; color: #c9a6ff; font-size: 1.2rem; }}
    .card p {{ margin: 0; color: #b4b4bf; font-size: 1rem; line-height: 1.55; }}
    .stat {{ text-align: center; }}
    .stat b {{ display: block; font-family: Georgia, serif; font-size: 2.2rem; color: #b57bff; }}
    .stat span {{ color: #a9a9b5; font-size: 0.98rem; }}
    .tri-red {{ border-left: 5px solid #ff3b30; }}
    .tri-orange {{ border-left: 5px solid #ff9500; }}
    .tri-green {{ border-left: 5px solid #34c759; }}
    .site-footer {{
        max-width: 1150px; margin: 4rem auto 1rem auto; padding-top: 1.6rem;
        border-top: 1px solid rgba(143, 44, 255, 0.3);
        text-align: center; color: #a9a9b5;
    }}
    .site-footer .names {{ color: #e6e6ea; font-weight: 700; font-size: 1.15rem; margin-top: 0.4rem; }}
    .site-footer .small {{ font-size: 0.85rem; margin-top: 0.8rem; color: #7d7d89; }}

    /* ---------- Section headings inside tabs ---------- */
    h2, h3, h4 {{ color: #f2f2f5; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: #9a9aa6; }}

    /* ---------- Metrics ---------- */
    [data-testid="stMetric"] {{
        background: rgba(18, 6, 34, 0.8);
        border: 1px solid rgba(143, 44, 255, 0.25);
        border-left: 4px solid {PURPLE};
        border-radius: 10px;
        padding: 12px 14px;
    }}

    /* ---------- Triage banners ---------- */
    .triage-critical, .triage-warning, .triage-ambulatory {{
        border-radius: 8px; padding: 10px 14px; font-weight: 600; margin-bottom: 10px;
    }}
    .triage-critical  {{ background: #ff3b3022; border: 1px solid #ff3b30; color: #ff6b64; }}
    .triage-warning   {{ background: #ff950022; border: 1px solid #ff9500; color: #ffb340; }}
    .triage-ambulatory{{ background: #34c75922; border: 1px solid #34c759; color: #5ce680; }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: #07030f;
        border-right: 1px solid rgba(143, 44, 255, 0.25);
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{ color: #b57bff; }}

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {{
        background: {PURPLE};
        color: #fff;
        border: none;
        border-radius: 8px;
        font-weight: 700;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{ background: #a65aff; color: #fff; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine(model_path: str):
    return ResQSenseEngine(model_path=model_path)


def find_col(df: pd.DataFrame, keyword: str):
    """Return the first column whose name contains `keyword` (case-insensitive)."""
    for c in df.columns:
        if keyword.lower() in str(c).lower():
            return c
    return None


def main():
    # ---------------- Site title ----------------
    st.markdown(
        """
        <div class="site-title">
            <h1>Human Detection in Disaster</h1>
            <p>AI Driven Disaster Management System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ================= SIDEBAR: CONTROL PANEL (click >> top-left to open) =================
    st.sidebar.title("Control Panel")

    st.sidebar.subheader("1. Active AI Model")
    existing_models = [DEFAULT_MODEL_NAME]
    if MODELS_DIR.exists():
        for p in MODELS_DIR.glob("*.pt"):
            if p.name not in existing_models:
                existing_models.append(p.name)

    selected_model_choice = st.sidebar.selectbox(
        "Select Detection Model:", existing_models, index=0,
        help="Choose standard YOLO or your custom-trained disaster model.",
    )
    uploaded_model_file = st.sidebar.file_uploader(
        "Upload Custom Model (.pt):", type=["pt"],
        help="Upload custom YOLO weights trained on disaster survivors or rubble.",
    )

    active_model_path = selected_model_choice
    if uploaded_model_file is not None:
        saved_custom_path = MODELS_DIR / uploaded_model_file.name
        with open(saved_custom_path, "wb") as f:
            f.write(uploaded_model_file.getbuffer())
        st.sidebar.success(f"Custom model saved: `{uploaded_model_file.name}`")
        active_model_path = str(saved_custom_path)

    st.sidebar.markdown("---")

    st.sidebar.subheader("2. Disaster Preprocessing")
    selected_filter = st.sidebar.selectbox(
        "Sensor / Enhancement Mode:", AVAILABLE_FILTERS, index=0,
        help="Simulates Thermal IR, CLAHE low-light penetration, or edge contrast.",
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader("3. Sensitivity & Thresholds")
    conf_thresh = st.sidebar.slider(
        "Confidence Threshold:", min_value=0.10, max_value=0.95,
        value=DEFAULT_CONF_THRESHOLD, step=0.05,
        help="Lower values detect occluded or partially buried victims.",
    )
    iou_thresh = st.sidebar.slider(
        "IoU Threshold (NMS):", min_value=0.20, max_value=0.80,
        value=DEFAULT_IOU_THRESHOLD, step=0.05,
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader("4. Sector & Alerts")
    grid_sector = st.sidebar.text_input("Active Sector:", "Sector-Alpha-1")
    enable_audio = st.sidebar.checkbox("Enable Audio Alert", value=True)

    # ---------------- Load engine ----------------
    try:
        engine = get_engine(active_model_path)
    except Exception as e:
        st.error(f"Failed to load AI model '{active_model_path}': {e}")
        return

    tracker = SurvivorTracker()
    logger = IncidentLogger()

    # ---------------- Navigation (the LAST tab is styled as the red pill) ----------------
    tab_home, tab_live, tab_media, tab_analytics, tab_logs, tab_research, tab_mission = st.tabs([
        "Home",
        "Live Mission",
        "Image Analysis",
        "Analytics",
        "Alerts",
        "Research",
        "Mission",
    ])

    # ================= HOME =================
    with tab_home:
        st.markdown(
            """
            <div class="hero">
                <h1>ResQ_sense</h1>
                <h2>AI-Powered Human Detection<br>for Disaster Response.</h2>
                <div class="tagline">Detect • Rescue • Save Lives.</div>
            </div>
            <div class="feature-card">
                <p>🚁 Drone - Live monitoring.</p>
                <p>🧍 Human Detection - YOLO AI.</p>
                <p>📍 GPS Tracking - Real-time.</p>
            </div>
            <div class="grid">
            <div class="card stat"><b>3</b><span>Triage levels</span></div>
            <div class="card stat"><b>5</b><span>Vision modes</span></div>
            <div class="card stat"><b>Live</b><span>Webcam detection</span></div>
            <div class="card stat"><b>CSV</b><span>Auditable mission log</span></div>
            </div>
            <div class="section-title">What it can do</div>
            <div class="grid">
            <div class="card"><h4>Live Webcam Detection</h4><p>Run the AI model on a webcam or USB camera in real time with FPS and latency shown on screen.</p></div>
            <div class="card"><h4>Custom YOLO Models</h4><p>Use the default YOLOv8 or upload your own trained .pt weights for rubble and disaster scenes.</p></div>
            <div class="card"><h4>Posture &amp; Triage</h4><p>Body shape tells us if a person is lying, crouched or standing, so the most urgent survivors come first.</p></div>
            <div class="card"><h4>Disaster Vision Filters</h4><p>Thermal Inferno, Ironbow, CLAHE low-light and Rubble Edge filters help in smoke, dust and darkness.</p></div>
            <div class="card"><h4>Survivor Tracking</h4><p>Every person gets a tracker ID, so the same survivor is not counted again and again.</p></div>
            <div class="card"><h4>Incident Logger</h4><p>Snapshots and a CSV record of every detection, ready to share with the rescue team.</p></div>
            </div>
            <div class="section-title">Triage colour code</div>
            <div class="grid">
            <div class="card tri-red"><h4>Critical</h4><p>Lying or trapped person. Highest priority, rescue first.</p></div>
            <div class="card tri-orange"><h4>Caution</h4><p>Crouched or partly hidden person. Needs a closer check.</p></div>
            <div class="card tri-green"><h4>Ambulatory</h4><p>Standing or walking survivor. Can be guided to safety.</p></div>
            </div>
            <div class="section-title">How it works</div>
            <div class="grid">
            <div class="card"><h4>1. Capture</h4><p>Take video from a webcam, or upload a drone or rubble photo.</p></div>
            <div class="card"><h4>2. Enhance</h4><p>Apply a thermal or low-light filter to see people more clearly.</p></div>
            <div class="card"><h4>3. Detect</h4><p>YOLO finds every human body and works out the posture.</p></div>
            <div class="card"><h4>4. Alert &amp; log</h4><p>Critical survivors raise an alert and are saved to the mission log.</p></div>
            </div>
            <div class="site-footer">
            <div>Designed by</div>
            <div class="names">Madhusudan Upadhyay &nbsp;•&nbsp; Abhay Sharma &nbsp;•&nbsp; Mihir Chaudhary</div>
            <div class="small">Human Detection in Disaster &nbsp;|&nbsp; AI Driven Disaster Management System</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ================= IMAGE ANALYSIS =================
    with tab_media:
        st.subheader("Disaster Drone / Rubble Image Analysis")
        st.caption("Upload reconnaissance photos, drone aerial snapshots, or rubble images to inspect survivors.")

        uploaded_img = st.file_uploader("Upload Image:", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_img is not None:
            image = Image.open(uploaded_img).convert("RGB")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Reconnaissance Image**")
                st.image(image, use_container_width=True)

            enhanced = apply_filter(frame, selected_filter)
            detections, lat_ms = engine.detect(enhanced, conf_threshold=conf_thresh, iou_threshold=iou_thresh)
            tracked = tracker.update(detections)

            for det in tracked:
                logger.log_detection(frame, det, grid_sector=grid_sector)

            hud_frame = engine.draw_hud(
                enhanced, tracked, latency_ms=lat_ms,
                filter_name=selected_filter, sector_name=grid_sector,
            )

            with col2:
                st.markdown(f"**AI Detection Overlay ({selected_filter})**")
                st.image(cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.info(f"Analysis complete: Found **{len(tracked)}** individuals with latency **{lat_ms:.1f} ms**.")

            if len(tracked) > 0:
                st.markdown("#### Detected Survivor Breakdown")
                det_records = []
                for det in tracked:
                    det_records.append({
                        "Tracker ID": f"ID #{det.get('tracker_id', 'N/A')}",
                        "Triage Level": det["triage_level"],
                        "Posture": det["posture"],
                        "Confidence": f"{det['confidence'] * 100:.1f}%",
                        "Aspect Ratio (W/H)": det["aspect_ratio"],
                        "Box (x1, y1, x2, y2)": str(det["box"]),
                    })
                st.dataframe(pd.DataFrame(det_records), use_container_width=True)

    # ================= ANALYTICS =================
    with tab_analytics:
        st.subheader("Mission Analytics")
        st.caption("Summary of everything logged by live streams and image analysis.")

        if INCIDENT_LOG_CSV.exists():
            df = pd.read_csv(INCIDENT_LOG_CSV)
        else:
            df = pd.DataFrame()

        if df.empty:
            st.info("No detections logged yet. Run a live stream or analyse an image to see analytics here.")
        else:
            posture_col = find_col(df, "posture")
            sector_col = find_col(df, "sector")
            id_col = find_col(df, "tracker")
            conf_col = find_col(df, "conf")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Log Entries", len(df))
            m2.metric("Unique Survivors", df[id_col].nunique() if id_col else "--")
            m3.metric("Sectors Covered", df[sector_col].nunique() if sector_col else "--")
            if conf_col is not None:
                conf_vals = pd.to_numeric(df[conf_col].astype(str).str.rstrip("%"), errors="coerce")
                m4.metric("Avg Confidence", f"{conf_vals.mean():.2f}")
            else:
                m4.metric("Avg Confidence", "--")

            c1, c2 = st.columns(2)
            if posture_col:
                with c1:
                    st.markdown("#### Detections by posture")
                    st.bar_chart(df[posture_col].value_counts())
            if sector_col:
                with c2:
                    st.markdown("#### Detections by sector")
                    st.bar_chart(df[sector_col].value_counts())

    # ================= ALERTS (incident log) =================
    with tab_logs:
        st.subheader("Search & Rescue Mission Log")
        st.caption("Auditable record of detected survivors, triage statuses, and snapshot crops.")

        col_export, col_refresh = st.columns([4, 1])
        with col_refresh:
            if st.button("Refresh Logs"):
                st.rerun()

        if INCIDENT_LOG_CSV.exists():
            df = pd.read_csv(INCIDENT_LOG_CSV)
            if not df.empty:
                st.dataframe(df.tail(50), use_container_width=True)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Mission Log (CSV)",
                    data=csv_bytes,
                    file_name=f"resq_sense_mission_log_{int(time.time())}.csv",
                    mime="text/csv",
                )
            else:
                st.info("Log is currently empty. Run live webcam or image analysis to log survivor detections.")
        else:
            st.info("No mission log file created yet.")

        st.markdown("#### Recent Snapshot Evidence Gallery")
        recent_crops = list(SNAPSHOTS_DIR.glob("*.jpg"))
        recent_crops.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        if recent_crops:
            gallery_cols = st.columns(4)
            for idx, snap_path in enumerate(recent_crops[:8]):
                with gallery_cols[idx % 4]:
                    st.image(str(snap_path), caption=snap_path.stem[:28], use_container_width=True)
        else:
            st.caption("No snapshots captured yet.")

    # ================= RESEARCH =================
    with tab_research:
        st.subheader("Scientific Basis & Framework Design")
        st.markdown(
            """
            This software is designed according to the framework outlined in:
            > **"Human Body Detection in Disaster Environments: A Review of Framework Design and Development Strategies"**  
            > *Vivek Sharma, Dr. Manish Dixit (September 2025)*  
            > *Journal of Global Research in Electronics and Communication, 1(9), 01-06.*

            ### Addressed Disaster Challenges
            1. **Variable Body Postures (Qi & Lou, 2023)**:
               - Trapped victims under rubble are frequently in a horizontal / prone posture.
               - The system evaluates bounding box geometry to flag **Lying/Trapped** survivors as highest triage priority (`CRITICAL / RED`).
            2. **Adverse Environmental Visibility (Ponzini et al. 2025, Kalyan et al. 2024)**:
               - Darkness, smoke, and dust are addressed via **Simulated Thermal (Inferno & Ironbow)** and **CLAHE Lightness Equalization**.
            3. **Real-time Edge Efficiency**:
               - Optimized stream processing capable of running at high FPS on portable edge devices and field rescue laptops.
            4. **Risk Communication Principles**:
               - **Accuracy & Timeliness**: Instantaneous detection and visual alerting.
               - **Clarity & Simplicity**: Plain, color-coded triage indicators without complex terminology.
               - **Transparency**: Auditable snapshot logging with coordinates and timestamps.
            """
        )

    # ================= MISSION (red button) =================
    with tab_mission:
        st.subheader("Current Mission Brief")
        st.caption("Open the sidebar (>> at the top-left) to change any of these settings.")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Active Sector", grid_sector)
        b2.metric("Model", str(active_model_path).split("\\")[-1].split("/")[-1])
        b3.metric("Vision Mode", selected_filter)
        b4.metric("Audio Alert", "ON" if enable_audio else "OFF")
        b5, b6, _, _ = st.columns(4)
        b5.metric("Confidence", f"{conf_thresh:.2f}")
        b6.metric("IoU (NMS)", f"{iou_thresh:.2f}")

    # ================= LIVE MISSION =================
    # NOTE: this block is intentionally LAST in the code. The live loop blocks the script,
    # so every other tab above is already drawn before streaming starts.
    with tab_live:
        st.subheader("Real-Time Webcam Human Body Detection")
        st.caption("AI model running on live feed with disaster triage & posture assessment.")

        ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1.4])
        cam_source_type = ctrl1.radio(
            "Video Feed Source:",
            ["Live Webcam (Native OpenCV)", "Browser Camera Snapshot"],
            index=0,
        )
        if cam_source_type == "Live Webcam (Native OpenCV)":
            cam_index = ctrl2.selectbox(
                "Select Camera Device:",
                options=[0, 1, 2],
                format_func=lambda x: f"Camera {x} (Built-in Webcam)" if x == 0 else f"Camera {x} (External USB Camera)",
                index=0,
            )
            run_btn = ctrl3.toggle("START LIVE STREAM", value=False)
        else:
            cam_index = 0
            run_btn = False

        telemetry_cols = st.columns(4)
        fps_metric = telemetry_cols[0].empty()
        latency_metric = telemetry_cols[1].empty()
        active_victims_metric = telemetry_cols[2].empty()
        triage_status_metric = telemetry_cols[3].empty()

        triage_banner_area = st.empty()
        feed_placeholder = st.empty()

        if cam_source_type == "Live Webcam (Native OpenCV)":
            if not run_btn:
                feed_placeholder.info(
                    "**Webcam stream on standby.**\n\n"
                    "Turn on **START LIVE STREAM** (top right of this tab) to begin detection."
                )
                fps_metric.metric("Stream FPS", "--")
                latency_metric.metric("Latency", "-- ms")
                active_victims_metric.metric("Active Survivors", "0")
                triage_status_metric.metric("Triage Status", "STANDBY")
            else:
                cap = cv2.VideoCapture(int(cam_index))
                if not cap.isOpened():
                    cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)

                if not cap.isOpened():
                    feed_placeholder.error(
                        f"Cannot access Camera {cam_index}.\n\n"
                        "Make sure **Camera 0 (Built-in Webcam)** is selected above, and that no other "
                        "app (Zoom, Teams, Windows Camera) is using the camera."
                    )
                else:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    prev_time = time.time()
                    fps_val = 0.0

                    try:
                        while run_btn:
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                feed_placeholder.warning("Frame read failed or camera disconnected.")
                                break

                            curr_t = time.time()
                            fps_val = 0.9 * fps_val + 0.1 * (1.0 / max(1e-5, (curr_t - prev_time)))
                            prev_time = curr_t

                            enhanced = apply_filter(frame, selected_filter)
                            detections, lat_ms = engine.detect(
                                enhanced, conf_threshold=conf_thresh, iou_threshold=iou_thresh,
                            )
                            tracked = tracker.update(detections)

                            for det in tracked:
                                logger.log_detection(frame, det, grid_sector=grid_sector)

                            hud_frame = engine.draw_hud(
                                enhanced, tracked, fps=fps_val, latency_ms=lat_ms,
                                filter_name=selected_filter, sector_name=grid_sector,
                            )

                            rgb_frame = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
                            feed_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                            fps_metric.metric("Stream FPS", f"{fps_val:.1f}")
                            latency_metric.metric("Latency", f"{lat_ms:.0f} ms")
                            active_victims_metric.metric("Active Survivors", len(tracked))

                            c_crit = sum(1 for d in tracked if d["priority_rank"] == 1)
                            c_warn = sum(1 for d in tracked if d["priority_rank"] == 2)
                            c_amb = sum(1 for d in tracked if d["priority_rank"] == 3)

                            if c_crit > 0:
                                triage_status_metric.metric("Triage Status", "CRITICAL")
                                triage_banner_area.markdown(
                                    f"<div class='triage-critical'>🚨 CRITICAL ALERT: {c_crit} Trapped/Lying survivor(s) detected!</div>",
                                    unsafe_allow_html=True,
                                )
                            elif c_warn > 0:
                                triage_status_metric.metric("Triage Status", "CAUTION")
                                triage_banner_area.markdown(
                                    f"<div class='triage-warning'>⚠️ CAUTION: {c_warn} Crouched/Partial survivor(s) in sector.</div>",
                                    unsafe_allow_html=True,
                                )
                            elif c_amb > 0:
                                triage_status_metric.metric("Triage Status", "AMBULATORY")
                                triage_banner_area.markdown(
                                    f"<div class='triage-ambulatory'>✅ AMBULATORY: {c_amb} Standing/Mobile survivor(s).</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                triage_status_metric.metric("Triage Status", "CLEAR")
                                triage_banner_area.empty()

                            time.sleep(0.01)
                    finally:
                        cap.release()

        else:
            camera_input_img = st.camera_input("Take Camera Snapshot for Analysis")
            if camera_input_img is not None:
                nparr = np.frombuffer(camera_input_img.getvalue(), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                enhanced = apply_filter(frame, selected_filter)
                detections, lat_ms = engine.detect(enhanced, conf_threshold=conf_thresh, iou_threshold=iou_thresh)
                tracked = tracker.update(detections)

                for det in tracked:
                    logger.log_detection(frame, det, grid_sector=grid_sector)

                hud_frame = engine.draw_hud(
                    enhanced, tracked, latency_ms=lat_ms,
                    filter_name=selected_filter, sector_name=grid_sector,
                )
                feed_placeholder.image(cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.success(f"Detected {len(tracked)} survivor(s) in {lat_ms:.1f}ms.")


if __name__ == "__main__":
    main()