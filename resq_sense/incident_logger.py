import csv
from datetime import datetime, timezone
from pathlib import Path
import cv2
from .config import INCIDENT_LOG_CSV, SNAPSHOTS_DIR

class IncidentLogger:
    def __init__(self):
        self.csv_path = INCIDENT_LOG_CSV
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    def log_detection(self, frame, detection, grid_sector="Unspecified"):
        now = datetime.now(timezone.utc).astimezone()
        stamp = now.strftime("%Y%m%d_%H%M%S_%f")
        x1, y1, x2, y2 = map(int, detection["box"])
        height, width = frame.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        crop = frame[y1:y2, x1:x2]
        snapshot_path = SNAPSHOTS_DIR / f"survivor_{stamp}.jpg"
        if crop.size:
            cv2.imwrite(str(snapshot_path), crop)

        row = {
            "timestamp": now.isoformat(timespec="seconds"),
            "sector": grid_sector,
            "tracker_id": detection.get("tracker_id", ""),
            "triage_level": detection.get("triage_level", ""),
            "posture": detection.get("posture", ""),
            "confidence": detection.get("confidence", ""),
            "box": str(detection.get("box", "")),
            "snapshot": str(snapshot_path) if crop.size else "",
        }
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return row
