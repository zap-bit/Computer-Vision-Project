"""
FastAPI service for reading backend detection logs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query


LOGS_DIR = Path("runs/backend_logs")
app = FastAPI(title="Inventory Backend API", version="0.1.0")


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
        raise HTTPException(status_code=404, detail=f"detections.db missing for run: {run_dir.name}")
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


@app.get("/latest-counts")
def latest_counts(run_id: str | None = Query(default=None)) -> dict[str, Any]:
    run_dir = _resolve_run_dir(run_id)
    conn = _open_db(run_dir)
    try:
        frame_row = conn.execute("SELECT MAX(frame_idx) AS frame_idx FROM detection_events").fetchone()
        latest_frame = frame_row["frame_idx"] if frame_row else None
        if latest_frame is None:
            return {"run_id": run_dir.name, "frame_idx": None, "counts": {}}

        rows = conn.execute(
            """
            SELECT class_name, count
            FROM detection_events
            WHERE frame_idx = ?
            ORDER BY class_name
            """,
            (latest_frame,),
        ).fetchall()
        counts = {row["class_name"]: int(row["count"]) for row in rows}
        non_zero_counts = {k: v for k, v in counts.items() if v > 0}
        return {"run_id": run_dir.name, "frame_idx": int(latest_frame), "counts": non_zero_counts}
    finally:
        conn.close()


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
        return {"run_id": run_dir.name, "class_name": class_name, "items": items}
    finally:
        conn.close()

