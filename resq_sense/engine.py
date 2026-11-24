
import time
import cv2
import numpy as np

from ultralytics import YOLO
from .triage import assess_posture


class ResQSenseEngine:
    """
    YOLO-based human detection engine.

    Priority labels are experimental visual heuristics,
    NOT medically validated triage classifications.
    """

    def __init__(
        self,
        model_path,
        device=None,
        image_size=640
    ):
        self.model_path = str(model_path)
        self.device = device
        self.image_size = int(image_size)

        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

        self.model = YOLO(self.model_path)

        if self.model is None:
            raise RuntimeError(
                f"Could not initialize YOLO model: {self.model_path}"
            )

    def detect(
        self,
        frame,
        conf_threshold=0.25,
        iou_threshold=0.45
    ):
        """
        Detect people in an image.

        Returns:
            detections: list of detection dictionaries
            latency_ms: inference duration in milliseconds
        """

        start = time.perf_counter()

        # Validate input image
        if frame is None:
            raise ValueError("Input frame is empty")

        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a NumPy array")

        if frame.size == 0:
            raise ValueError("Input frame contains no image data")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Expected a three-channel BGR image"
            )

        # Validate thresholds
        conf_threshold = float(conf_threshold)
        iou_threshold = float(iou_threshold)

        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError(
                "Confidence threshold must be between 0 and 1"
            )

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(
                "IoU threshold must be between 0 and 1"
            )

        detections = []

        # Run YOLO inference
        prediction_args = {
            "source": frame,
            "conf": conf_threshold,
            "iou": iou_threshold,
            "classes": [0],
            "imgsz": self.image_size,
            "verbose": False
        }

        if self.device is not None:
            prediction_args["device"] = self.device

        results = self.model.predict(**prediction_args)

        if results:
            result = results[0]
            boxes = result.boxes
            names = result.names

            if boxes is not None:
                for box in boxes:

                    # Extract bounding box
                    coords = (
                        box.xyxy[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    if len(coords) != 4:
                        continue

                    x1, y1, x2, y2 = map(
                        float, coords
                    )

                    # Reject invalid coordinates
                    if not np.all(np.isfinite(
                        [x1, y1, x2, y2]
                    )):
                        continue

                    height, width = frame.shape[:2]

                    x1 = max(0, min(width - 1, int(x1)))
                    y1 = max(0, min(height - 1, int(y1)))
                    x2 = max(0, min(width - 1, int(x2)))
                    y2 = max(0, min(height - 1, int(y2)))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    confidence = float(
                        box.conf[0].detach().cpu().item()
                    )

                    class_id = int(
                        box.cls[0].detach().cpu().item()
                    )

                    class_name = (
                        names.get(class_id, str(class_id))
                        if isinstance(names, dict)
                        else str(class_id)
                    )

                    # Experimental posture heuristic
                    posture, triage = assess_posture(
                        [x1, y1, x2, y2]
                    )

                    box_width = x2 - x1
                    box_height = y2 - y1

                    aspect_ratio = round(
                        box_width / max(1, box_height),
                        3
                    )

                    detections.append({
                        "box": [
                            x1, y1, x2, y2
                        ],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name,
                        "posture": posture,
                        "triage_level": triage,
                        "aspect_ratio": aspect_ratio
                    })

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        return detections, latency_ms

    def draw_hud(
        self,
        frame,
        detections,
        latency_ms=0.0,
        filter_name="Normal",
        sector_name=""
    ):

        output = frame.copy()

        colors = {
            "CRITICAL": (48, 59, 255),
            "WARNING": (0, 149, 255),
            "AMBULATORY": (52, 199, 52)
        }

        for det in detections:

            x1, y1, x2, y2 = map(
                int, det["box"]
            )

            level = det.get(
                "triage_level", "WARNING"
            )

            color = colors.get(
                level, (255, 255, 0)
            )

            label = (
                f'ID {det.get("tracker_id", "-")} '
                f'{level} '
                f'{det.get("confidence", 0):.2f}'
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        cv2.putText(
            output,
            f"{sector_name} | {filter_name} | "
            f"{latency_ms:.1f} ms",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return output
