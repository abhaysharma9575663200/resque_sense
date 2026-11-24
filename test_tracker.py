
from resq_sense.tracker import SurvivorTracker

tracker = SurvivorTracker()

frame_1 = [
    {"box": [100, 100, 150, 200], "confidence": 0.9}
]

frame_2 = [
    {"box": [105, 102, 155, 202], "confidence": 0.92}
]

result_1 = tracker.update(frame_1)
result_2 = tracker.update(frame_2)

print("Frame 1:", result_1)
print("Frame 2:", result_2)

assert result_1[0]["tracker_id"] == result_2[0]["tracker_id"]

print("SUCCESS: Tracker preserved the survivor ID.")
