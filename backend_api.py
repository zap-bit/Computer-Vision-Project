"""
FastAPI service for backend detection logs and live YOLO camera feed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

LOGS_DIR = Path("runs/backend_logs")

app = FastAPI(title="Inventory Backend API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = YOLO("segment/runs/detect/shelf_inventory/weights/best.engine")
    print("Loaded TensorRT model")
except Exception:
    model = YOLO("segment/runs/detect/shelf_inventory/weights/best.pt")
    print("Loaded PyTorch model")

ALLOWED_CLASSES = {
    "chocolate",
    "granola",
    "mouse",
    "pen",
    "waterbottle",
}

live_counts = {item: 0 for item in ALLOWED_CLASSES}


def _resolve_run_dir(run_id: str | None) -> Path:
    if run_id:
        run_dir = LOGS_DIR / run_id
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return run_dir

    if not LOGS_DIR.exists():
        raise HTTPException(status_code=404, detail="No backend log directory found.")

    run_dirs = [p for p in LOGS_DIR.iterdir() if p.is_dir()]
    if not run_dirs:
        raise HTTPException(status_code=404, detail="No runs found.")

    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _open_db(run_dir: Path) -> sqlite3.Connection:
    db_path = run_dir / "detections.db"

    if not db_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"detections.db missing for run: {run_dir.name}",
        )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runs/latest")
def latest_run() -> dict[str, str]:
    run_dir = _resolve_run_dir(run_id=None)
    return {"run_id": run_dir.name}


def generate_camera_frames():
    global live_counts

    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()

        if not success:
            continue

        results = model.predict(source=frame, conf=0.25, verbose=False)

        counts = {item: 0 for item in ALLOWED_CLASSES}

        for result in results:
            names = result.names

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = names[cls_id].lower()

                if label not in ALLOWED_CLASSES:
                    continue

                counts[label] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        live_counts = counts

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

    camera.release()


@app.get("/video-feed")
def video_feed():
    return StreamingResponse(
        generate_camera_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/latest-counts")
def latest_counts(run_id: str | None = Query(default=None)) -> dict[str, Any]:
    return {
        "run_id": "live_camera",
        "frame_idx": None,
        "counts": live_counts,
    }


@app.get("/trend/{class_name}")
def class_trend(
    class_name: str,
    run_id: str | None = Query(default=None),
    limit: int = Query(default=120, ge=1, le=5000),
) -> dict[str, Any]:
    run_dir = _resolve_run_dir(run_id)
    conn = _open_db(run_dir)

    try:
        rows = conn.execute(
            """
            SELECT timestamp_utc, frame_idx, count
            FROM detection_events
            WHERE class_name = ?
            ORDER BY frame_idx DESC
            LIMIT ?
            """,
            (class_name, limit),
        ).fetchall()

        items = [
            {
                "timestamp_utc": row["timestamp_utc"],
                "frame_idx": int(row["frame_idx"]),
                "count": int(row["count"]),
            }
            for row in reversed(rows)
        ]

        return {
            "run_id": run_dir.name,
            "class_name": class_name,
            "items": items,
        }

    finally:
        conn.close()