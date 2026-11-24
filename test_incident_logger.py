
import csv
import tempfile

import cv2
import numpy as np

from pathlib import Path
from unittest.mock import patch

from resq_sense.incident_logger import IncidentLogger


# Create a temporary folder for testing.
with tempfile.TemporaryDirectory() as temp_dir:

    temp_path = Path(temp_dir)

    csv_path = temp_path / "incident_log.csv"
    snapshots_path = temp_path / "snapshots"

    # Redirect logger output to temporary locations.
    with patch(
        "resq_sense.incident_logger.INCIDENT_LOG_CSV",
        csv_path,
    ), patch(
        "resq_sense.incident_logger.SNAPSHOTS_DIR",
        snapshots_path,
    ):

        logger = IncidentLogger()

        # Simulate a camera frame.
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Simulate one detected survivor.
        detection = {
            "box": [100, 100, 250, 350],
            "confidence": 0.95,
            "tracker_id": 1,
            "posture": "Lying / Horizontal",
            "triage_level": "CRITICAL",
        }

        result = logger.log_detection(
            frame,
            detection,
            grid_sector="Sector A",
        )

        print("Incident recorded:", result)

        # Verify CSV.
        assert csv_path.exists()

        with csv_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            rows = list(csv.DictReader(file))

        assert len(rows) == 1
        assert rows[0]["tracker_id"] == "1"
        assert rows[0]["sector"] == "Sector A"

        # Verify evidence snapshot.
        snapshot = Path(result["snapshot"])

        assert snapshot.exists()

        saved_image = cv2.imread(str(snapshot))

        assert saved_image is not None
        assert saved_image.shape[:2] == (250, 150)

        print("\nSUCCESS: Incident logging test passed.")
        print("CSV logging: OK")
        print("Evidence snapshot: OK")
        print("Bounding-box cropping: OK")
