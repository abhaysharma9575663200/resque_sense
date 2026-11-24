
import cv2

from resq_sense.engine import ResQSenseEngine
from resq_sense.tracker import SurvivorTracker
from resq_sense.triage import assess_posture

print("========== RESQ SENSE INTEGRATION TEST ==========")

# 1. Load image
image = cv2.imread("test_image.jpg")

if image is None:
    raise FileNotFoundError("Cannot load test_image.jpg")

print("[OK] Image loaded")

# 2. Initialize engine
engine = ResQSenseEngine("yolov8n.pt")

print("[OK] Detection engine initialized")

# 3. Initialize tracker
tracker = SurvivorTracker()

print("[OK] Survivor tracker initialized")

# 4. Run detection
detections, latency = engine.detect(image)

print("[OK] Detection completed")

# 5. Track detections
tracked = tracker.update(detections)

print("[OK] Tracking completed")

# 6. Verify output structure
required_keys = [
    "box",
    "confidence",
    "class_name",
    "posture",
    "triage_level",
    "aspect_ratio",
    "tracker_id",
]

for detection in tracked:
    for key in required_keys:
        assert key in detection, f"Missing key: {key}"

    posture, priority = assess_posture(detection["box"])

    assert detection["posture"] == posture
    assert detection["triage_level"] == priority

print("\nDetected people:", len(tracked))
print("Inference time:", round(latency, 2), "ms")

for detection in tracked:
    print(
        "ID:", detection["tracker_id"],
        "| Posture:", detection["posture"],
        "| Priority:", detection["triage_level"],
        "| Confidence:", round(detection["confidence"], 3)
    )

print("\nSUCCESS: Backend integration test passed.")
