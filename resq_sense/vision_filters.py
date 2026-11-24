import cv2
import numpy as np
from .config import FILTER_NORMAL

def apply_filter(frame, filter_name=FILTER_NORMAL):
    """Apply an optional visibility enhancement to a BGR OpenCV frame."""
    if frame is None:
        return frame
    if filter_name == "CLAHE Low-Light":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    if filter_name in ("Thermal Inferno", "Thermal Ironbow"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        palette = cv2.COLORMAP_INFERNO if filter_name == "Thermal Inferno" else cv2.COLORMAP_HOT
        return cv2.applyColorMap(gray, palette)
    if filter_name == "Edge Contrast":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return frame.copy()
