"""
Backend storage layer for detection events.

Writes detection events to both:
- CSV (easy inspection)
- SQLite (queryable for dashboard/API)
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class BackendStorage:
    """Persist per-class detection counts for each logged frame."""

    def __init__(self, run_id: str, base_dir: Path) -> None:
        self.run_id = run_id
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.base_dir / "detection_events.csv"
        self.db_path = self.base_dir / "detections.db"

        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(
            [
                "timestamp_utc",
                "run_id",
                "frame_idx",
                "elapsed_seconds",
                "class_name",
                "count",
                "inference_ms",
                "fps",
            ]
        )

        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                run_id TEXT NOT NULL,
                frame_idx INTEGER NOT NULL,
                elapsed_seconds REAL NOT NULL,
                class_name TEXT NOT NULL,
                count INTEGER NOT NULL,
                inference_ms REAL NOT NULL,
                fps REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_detection_events_run_frame "
            "ON detection_events(run_id, frame_idx)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_detection_events_class_time "
            "ON detection_events(class_name, timestamp_utc)"
        )
        self._conn.commit()

    def log_counts(
        self,
        frame_idx: int,
        elapsed_seconds: float,
        counts: dict[str, int],
        inference_ms: float,
        fps: float,
    ) -> None:
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        rows: list[tuple[str, str, int, float, str, int, float, float]] = []

        for class_name, count in counts.items():
            rows.append(
                (
                    timestamp_utc,
                    self.run_id,
                    frame_idx,
                    round(elapsed_seconds, 3),
                    class_name,
                    int(count),
                    round(inference_ms, 3),
                    round(fps, 3),
                )
            )

        self._writer.writerows(rows)
        self._conn.executemany(
            """
            INSERT INTO detection_events (
                timestamp_utc, run_id, frame_idx, elapsed_seconds,
                class_name, count, inference_ms, fps
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def get_latest_counts(self) -> dict[str, int]:
        frame_row = self._conn.execute(
            "SELECT MAX(frame_idx) FROM detection_events WHERE run_id = ?",
            (self.run_id,),
        ).fetchone()
        latest_frame = frame_row[0]
        if latest_frame is None:
            return {}

        rows = self._conn.execute(
            """
            SELECT class_name, count
            FROM detection_events
            WHERE run_id = ? AND frame_idx = ?
            """,
            (self.run_id, latest_frame),
        ).fetchall()
        return {name: int(count) for name, count in rows}

    def get_class_trend(self, class_name: str, limit: int = 120) -> list[dict[str, float | int | str]]:
        rows = self._conn.execute(
            """
            SELECT timestamp_utc, frame_idx, count
            FROM detection_events
            WHERE run_id = ? AND class_name = ?
            ORDER BY frame_idx DESC
            LIMIT ?
            """,
            (self.run_id, class_name, limit),
        ).fetchall()
        rows.reverse()
        return [
            {"timestamp_utc": ts, "frame_idx": int(frame_idx), "count": int(count)}
            for ts, frame_idx, count in rows
        ]

    def close(self) -> None:
        self._csv_file.flush()
        self._csv_file.close()
        self._conn.commit()
        self._conn.close()

