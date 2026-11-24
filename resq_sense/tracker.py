
class SurvivorTracker:
    """
    Lightweight IoU-based survivor tracker.

    IDs are temporary tracking labels, not personal identities.
    """

    def __init__(self, iou_threshold=0.3, max_disappeared=10):
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be between 0 and 1")

        if max_disappeared < 0:
            raise ValueError("max_disappeared cannot be negative")

        self.iou_threshold = float(iou_threshold)
        self.max_disappeared = int(max_disappeared)
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    def update(self, detections):
        detections = [dict(d) for d in detections]

        # Build possible matches between existing tracks and detections.
        candidates = []

        for track_id, track in self.tracks.items():
            for index, detection in enumerate(detections):
                score = self._iou(track["box"], detection["box"])

                if score >= self.iou_threshold:
                    candidates.append((score, track_id, index))

        # Match highest IoU pairs first.
        candidates.sort(reverse=True)

        matched_tracks = set()
        matched_detections = set()
        output = [None] * len(detections)

        for score, track_id, index in candidates:
            if track_id in matched_tracks:
                continue

            if index in matched_detections:
                continue

            detection = detections[index]
            detection["tracker_id"] = track_id

            self.tracks[track_id] = {
                "box": detection["box"],
                "disappeared": 0,
            }

            matched_tracks.add(track_id)
            matched_detections.add(index)
            output[index] = detection

        # Assign new IDs to unmatched detections.
        for index, detection in enumerate(detections):
            if index not in matched_detections:
                track_id = self.next_id
                self.next_id += 1

                detection["tracker_id"] = track_id

                self.tracks[track_id] = {
                    "box": detection["box"],
                    "disappeared": 0,
                }

                output[index] = detection

        # Age tracks that were not matched.
        for track_id in list(self.tracks):
            if track_id not in matched_tracks:
                if track_id not in {
                    detection["tracker_id"] for detection in output
                    if detection is not None
                }:
                    self.tracks[track_id]["disappeared"] += 1

                    if (
                        self.tracks[track_id]["disappeared"]
                        > self.max_disappeared
                    ):
                        del self.tracks[track_id]

        return output
