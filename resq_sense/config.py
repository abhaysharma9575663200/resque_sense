from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_NAME = str(ROOT_DIR / "yolov8n.pt")
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

FILTER_NORMAL = "Normal"
AVAILABLE_FILTERS = [
    FILTER_NORMAL,
    "CLAHE Low-Light",
    "Thermal Inferno",
    "Thermal Ironbow",
    "Edge Contrast",
]

MODELS_DIR = ROOT_DIR / "models"
SNAPSHOTS_DIR = ROOT_DIR / "snapshots"
DATA_DIR = ROOT_DIR / "data"
INCIDENT_LOG_CSV = DATA_DIR / "incident_log.csv"

COLOR_CRITICAL_HEX = "#ff3b30"
COLOR_WARNING_HEX = "#ff9500"
COLOR_AMBULATORY_HEX = "#34c759"

for directory in (MODELS_DIR, SNAPSHOTS_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)
