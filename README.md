# ResQ_sense: AI-Powered Disaster Search & Rescue Human Detection System

[![Research Basis](https://img.shields.io/badge/Research-JGREC%20Sept%202025-blue.svg)](https://jgrec.info)
[![Python](https://img.shields.io/badge/Python-3.13%2B-green.svg)](https://python.org)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8%20%7C%20YOLO11-red.svg)](https://github.com/ultralytics/ultralytics)

**ResQ_sense** is an AI-powered human body detection and triage system designed for search-and-rescue (SAR) operations in catastrophic disaster environments (earthquakes, building collapses, landslides, floods).

It directly operationalizes the architectural principles and findings presented in:  
> **"Human Body Detection in Disaster Environments: A Review of Framework Design and Development Strategies"**  
> *Vivek Sharma & Dr. Manish Dixit (September 2025)*  
> *Journal of Global Research in Electronics and Communication (JGREC), 1(9), 01–06.*

---

## 🌟 Key Features

### 1. 🎥 Live Webcam Model Execution
- Run your AI model **directly on a live webcam feed** in real time.
- Zero-latency native OpenCV rendering (`app_live.py`) and interactive mission control dashboard (`app_dashboard.py`).

### 2. 🧠 Custom Model Flexibility
- Built to run **your own model weights** (`.pt`) alongside default YOLO models (`yolov8n.pt`, `yolov8s.pt`, etc.).
- Drop your weights in the `models/` directory or upload directly through the web UI.

### 3. 🚨 Posture & Disaster Triage Assessment
*(Based on Qi & Lou, 2023 / Sharma & Dixit, 2025)*
- Automatically classifies survivor body posture and assigns emergency triage priority:
  - 🟥 **[CRITICAL / TRAPPED]**: Prone / horizontal body ($W/H \ge 1.15$), indicating trapped, fallen, or immobilized victims under rubble.
  - 🟧 **[CAUTION / PARTIAL]**: Crouched or partially occluded individuals.
  - 🟩 **[AMBULATORY / UPRIGHT]**: Standing or walking survivors.

### 4. 🎛️ Disaster Vision Preprocessing (Multimodal Simulation)
*(Based on Ponzini et al., 2025; Kalyan et al., 2024)*
- **Simulated Thermal (Inferno & Ironbow/JET)**: Generates false-color thermal heatmaps from optical feeds, isolating human thermal gradients.
- **Low-Light CLAHE**: Penetrates dark collapsed structures, night operations, and light dust/smoke.
- **Rubble Edge Filter**: Sharpens concrete and debris fracture lines to distinguish human forms from rubble.

### 5. 📋 Auditable Mission Incident Logger
- Auto-saves cropped evidence snapshots of detected survivors into `logs/snapshots/`.
- Appends auditable records to `logs/rescue_mission_log.csv` (Timestamp, Tracker ID, Posture, Confidence, Sector, Snapshot filename).
- One-click CSV export for rescue dispatch coordination.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Webcam or USB camera connected

### Installation
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running ResQ_sense

### Mode 1: High-Performance Live Webcam HUD (Recommended for Real-Time Field Ops)
Launch the native OpenCV HUD:
```bash
python app_live.py
```
Or with custom arguments:
```bash
python app_live.py --source 0 --model yolov8n.pt --conf 0.35 --sector Sector-Bravo-2
```

#### Native Keyboard Shortcuts:
| Key | Action |
|---|---|
| `[T]` | Cycle disaster vision filters (Normal -> Thermal Inferno -> Ironbow -> CLAHE -> Rubble Edge) |
| `[S]` | Save manual high-resolution evidence snapshot |
| `[+]` / `[-]` | Increase / Decrease detection confidence threshold in real time |
| `[Q]` / `[ESC]` | Quit application |

---

### Mode 2: Mission Control Web Dashboard (Streamlit UI)
Launch the interactive mission dashboard:
```bash
streamlit run app_dashboard.py
```
This gives you:
- Live webcam streaming with slider adjustments
- Custom model `.pt` drag-and-drop uploader
- Photo and reconnaissance footage analysis
- Live victim gallery and CSV mission report downloader

---

## 📁 Repository Structure

```
Insaan_finder/
│
├── resq_sense/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Global constants, paths, triage thresholds
│   ├── engine.py             # Ultralytics YOLO loader & tactical HUD renderer
│   ├── vision_filters.py     # Thermal, CLAHE, and debris edge enhancements
│   ├── triage.py             # Posture classification & urgency badge logic
│   ├── tracker.py            # Centroid tracker for unique survivor counting
│   └── incident_logger.py    # Snapshot saving and CSV audit logger
│
├── models/                   # Folder for your custom trained weights (.pt)
│   └── README.md
├── logs/                     # Saved snapshots and CSV rescue logs
│   ├── snapshots/
│   └── rescue_mission_log.csv
│
├── app_live.py               # Native high-FPS live webcam HUD runner
├── app_dashboard.py          # Streamlit Mission Control Web Dashboard
├── requirements.txt          # Python dependencies
└── README.md                 # System documentation
```

---

## 🔬 Research Citation
```bibtex
@article{sharma2025human,
  title={Human Body Detection in Disaster Environments: A Review of Framework Design and Development Strategies},
  author={Sharma, Vivek and Dixit, Manish},
  journal={Journal of Global Research in Electronics and Communication},
  volume={1},
  number={9},
  pages={01--06},
  year={2025}
}
```
