# Backend Runbook (Computer Vision Project)

This document covers everything implemented so far for backend work:

- dataset preparation script
- inference logging pipeline
- CSV + SQLite storage
- FastAPI endpoints for frontend/dashboard
- test and troubleshooting commands

---

## 1) Project Files Added/Updated

### Added
- `dataset_setup.py`
- `inventory_logger.py`
- `backend_storage.py`
- `inference_backend.py`
- `backend_api.py`

### Updated
- `requirement.txt` (added backend dependencies)

---

## 2) Environment Setup

Run from project root (`homework5_programming`):

```powershell
python -m pip install -r requirement.txt
```

If needed, direct installs used during development:

```powershell
python -m pip install ultralytics opencv-python fastapi uvicorn httpx
```

---

## 3) Dataset Preparation (YOLO Format)

Use `dataset_setup.py` to:
- validate image/label pairs
- split into train/val/test
- generate `dataset.yaml`

### Expected input
- one folder with images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`)
- one folder with YOLO label `.txt` files
- matching base names (example: `img_01.jpg` and `img_01.txt`)

### Command

```powershell
python dataset_setup.py `
  --images-dir .\raw\images `
  --labels-dir .\raw\labels `
  --output-dir .\data `
  --classes water_bottle orange apple `
  --train-ratio 0.8 `
  --val-ratio 0.2 `
  --seed 42
```

### Output
- `data/images/train`, `data/images/val`, `data/images/test`
- `data/labels/train`, `data/labels/val`, `data/labels/test`
- `data/dataset.yaml`

---

## 4) Inference + Backend Logging

Use `inference_backend.py` for live webcam inference with persistence.

### Command (normal run)

```powershell
python inference_backend.py --model yolov8n.pt --camera-index 0 --run-name shelf_inventory
```

### Useful options
- `--conf 0.3` confidence threshold
- `--log-interval 2` log every N frames
- `--max-frames 100` auto-stop after N frames (good for testing)
- `--camera-index 0` camera id

### Example test run

```powershell
python inference_backend.py `
  --model yolov8n.pt `
  --camera-index 0 `
  --run-name quick_test `
  --log-interval 2 `
  --max-frames 20
```

### Keyboard controls
- `q` quit
- `s` save screenshot

### Output folder per run

`runs/backend_logs/<run_id>/`

Contains:
- `frame_counts.csv` (frame-level wide log)
- `session_summary.json` (summary metrics)
- `detection_events.csv` (normalized event rows)
- `detections.db` (SQLite database)

---

## 5) Storage Schema

SQLite table: `detection_events`

Columns:
- `timestamp_utc`
- `run_id`
- `frame_idx`
- `elapsed_seconds`
- `class_name`
- `count`
- `inference_ms`
- `fps`

This is the backend source for API queries and frontend trend charts.

---

## 6) API Service (FastAPI)

`backend_api.py` serves run data from `runs/backend_logs`.

### Start API

```powershell
uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
```

### Endpoints
- `GET /health`
- `GET /runs/latest`
- `GET /latest-counts?run_id=<optional>`
- `GET /trend/{class_name}?run_id=<optional>&limit=120`
- `GET /docs` (Swagger UI)

### Quick checks in browser
- <http://127.0.0.1:8000/docs>
- <http://127.0.0.1:8000/health>
- <http://127.0.0.1:8000/runs/latest>
- <http://127.0.0.1:8000/latest-counts>
- <http://127.0.0.1:8000/trend/apple?limit=20>

Note: `GET /` currently returns 404 (expected). Use `/docs` for UI.

---

## 7) Quick Validation Commands

### Syntax checks

```powershell
python -m py_compile dataset_setup.py inventory_logger.py backend_storage.py inference_backend.py backend_api.py
```

### Dataset script help

```powershell
python dataset_setup.py --help
```

### Inference script help

```powershell
python inference_backend.py --help
```

### API endpoint smoke test

```powershell
python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
```

---

## 8) How to Stop Running Services

### Stop API server
- In the terminal running `uvicorn`, press `Ctrl + C`

### Stop inference script
- In the inference window, press `q`
- or in terminal press `Ctrl + C` (clean shutdown is supported)

### If camera appears stuck

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' } | Select-Object ProcessId,CommandLine
Stop-Process -Id <PID> -Force
```

Then confirm camera can open again:

```powershell
python -c "import cv2; cap=cv2.VideoCapture(0); ok=cap.isOpened(); ret,frame=(cap.read() if ok else (False,None)); print(f'camera_open={ok} frame_read={ret}'); cap.release()"
```

---

## 9) Recommended Team Workflow (Backend)

1. Collect/labeled data from team.
2. Run `dataset_setup.py` to build clean dataset structure.
3. Run `inference_backend.py` to generate logs and DB.
4. Start `backend_api.py` and give frontend `/latest-counts` + `/trend` endpoints.
5. Use stored run artifacts for evaluation/demo reports.

---

## 10) Next Backend Tasks

- add root endpoint (`GET /`) for friendly API landing page
- add filtering to `/latest-counts` (include/exclude zero values)
- add endpoint to list all run IDs
- optionally package backend as `backend/` folder for cleaner repo layout

