def assess_posture(box):
    """Heuristic posture estimate from bounding-box width/height; not medical triage."""
    x1, y1, x2, y2 = map(float, box)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    ratio = width / height
    if ratio >= 1.20:
        return "Lying / Horizontal", "CRITICAL"
    if ratio >= 0.75:
        return "Crouched / Partial", "WARNING"
    return "Standing / Vertical", "AMBULATORY"
