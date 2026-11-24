"""
ResQ_sense: Live Webcam Tactical HUD Runner (High-Performance Native Mode).
Provides zero-latency real-time video stream analysis from webcam or video source.

Keyboard Controls:
  [T]       - Cycle Disaster Vision Filters (Normal, Thermal Inferno, Ironbow, CLAHE, Rubble Edge)
  [S]       - Save Evidence Snapshot manually
  [+] / [-] - Increase / Decrease Detection Confidence Threshold
  [C]       - Switch Camera Index (0 -> 1 -> ...)
  [Q] / ESC - Quit Application
"""

import sys
import time
import argparse
from pathlib import Path
import cv2

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from resq_sense.config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    AVAILABLE_FILTERS,
    FILTER_NORMAL,
    MODELS_DIR,
)
from resq_sense.engine import ResQSenseEngine
from resq_sense.vision_filters import apply_filter
from resq_sense.tracker import SurvivorTracker
from resq_sense.incident_logger import IncidentLogger

# Optional Windows beep alert
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


def trigger_alert(sound: bool = True):
    if sound and HAS_WINSOUND:
        try:
            winsound.Beep(1200, 150)
        except Exception:
            pass


def run_live_feed(
    source: int = 0,
    model_path: str = DEFAULT_MODEL_NAME,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    enable_audio_alert: bool = True,
    sector_name: str = "Alpha-1",
    record_output: str = None,
):
    print("=" * 65)
    print("  ResQ_sense: AI Disaster Human Body Detection (Live HUD)")
    print("=" * 65)
    print(f"[*] Initializing camera source: {source}")
    print(f"[*] Loading detection model: {model_path}")
    print(f"[*] Initial confidence threshold: {conf_threshold:.2f}")

    # Initialize Engine
    engine = ResQSenseEngine(model_path=model_path)
    tracker = SurvivorTracker()
    logger = IncidentLogger()

    # Open Camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[!] Warning: Unable to open video source '{source}'.")
        print("[*] Attempting fallback to DirectShow backend or source 0...")
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[ERROR] Cannot access webcam. Please verify webcam connection.")
            return

    # Configure optimal camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    filter_index = 0
    current_filter = AVAILABLE_FILTERS[filter_index]

    prev_time = time.time()
    fps = 0.0

    video_writer = None
    if record_output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(record_output, fourcc, 20.0, (w, h))

    print("\n[+] System Active! Press [Q] to quit, [T] to toggle thermal/vision filters, [S] for snapshot.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[!] Frame read failed or end of stream.")
                break

            # Calculate FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
            prev_time = curr_time

            # 1. Apply Disaster Vision Enhancement Filter
            enhanced_frame = apply_filter(frame, current_filter)

            # 2. Run AI Human Body Detection & Triage
            detections, latency_ms = engine.detect(
                enhanced_frame,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

            # 3. Update Centroid Tracker
            tracked_detections = tracker.update(detections)

            # 4. Check for newly detected critical/lying victims & log incidents
            has_critical = any(d["priority_rank"] == 1 for d in tracked_detections)
            for det in tracked_detections:
                snapshot_file = logger.log_detection(frame, det, grid_sector=sector_name)
                if snapshot_file and det["priority_rank"] == 1:
                    print(f"\n[ALERT - CRITICAL SURVIVOR] Tracker {det.get('tracker_id')} | Posture: {det['posture']} | Snapshot: {snapshot_file}")
                    trigger_alert(enable_audio_alert)

            # 5. Render Tactical HUD
            display_frame = engine.draw_hud(
                enhanced_frame,
                tracked_detections,
                fps=fps,
                latency_ms=latency_ms,
                filter_name=current_filter,
                sector_name=sector_name,
            )

            # Record if enabled
            if video_writer:
                video_writer.write(display_frame)

            # Display window
            cv2.imshow("ResQ_sense - Disaster Search & Rescue HUD", display_frame)

            # Key Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('t') or key == ord('T'):
                filter_index = (filter_index + 1) % len(AVAILABLE_FILTERS)
                current_filter = AVAILABLE_FILTERS[filter_index]
                print(f"[*] Filter switched to: {current_filter}")
            elif key == ord('s') or key == ord('S'):
                manual_snap = logger.snapshots_dir / f"manual_snap_{int(time.time())}.jpg"
                cv2.imwrite(str(manual_snap), display_frame)
                print(f"[+] Saved snapshot to: {manual_snap}")
            elif key == ord('+') or key == ord('='):
                conf_threshold = min(0.95, round(conf_threshold + 0.05, 2))
                print(f"[*] Confidence threshold: {conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                conf_threshold = max(0.10, round(conf_threshold - 0.05, 2))
                print(f"[*] Confidence threshold: {conf_threshold:.2f}")

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("[*] ResQ_sense session terminated safely.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResQ_sense Live Webcam Disaster Human Detector")
    parser.add_argument("--source", default=0, help="Webcam index (0, 1...) or video file path")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model weights path (yolov8n.pt or custom .pt)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD, help="NMS IoU threshold")
    parser.add_argument("--sector", default="Sector-Alpha-1", help="Mission grid sector identifier")
    parser.add_argument("--no-sound", action="store_true", help="Disable audio alerts")
    parser.add_argument("--record", default=None, help="Optional output MP4 video path to record session")

    args = parser.parse_args()

    # Convert source to int if numeric
    src = int(args.source) if str(args.source).isdigit() else args.source

    run_live_feed(
        source=src,
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        enable_audio_alert=not args.no_sound,
        sector_name=args.sector,
        record_output=args.record,
    )
