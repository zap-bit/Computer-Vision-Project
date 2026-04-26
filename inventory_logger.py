"""
Backend utility for persisting detection counts during inference.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


class InventoryLogger:
    """Write frame-level detection counts and session-level summary."""

    def __init__(self, run_name: str = "shelf_inventory", output_dir: str = "runs/backend_logs") -> None:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{run_name}_{timestamp}"
        self.base_dir = Path(output_dir) / self.run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.base_dir / "frame_counts.csv"
        self.summary_path = self.base_dir / "session_summary.json"

        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)
        self._header_written = False
        self._class_names: list[str] = []

        self._frame_count = 0
        self._fps_values: list[float] = []
        self._inference_times_ms: list[float] = []
        self._max_counts: dict[str, int] = {}

    def log_frame(self, elapsed_seconds: float, counts: dict[str, int], fps: float, inference_ms: float) -> None:
        if not self._header_written:
            self._class_names = sorted(counts.keys())
            self._writer.writerow(["frame_idx", "elapsed_seconds", "fps", "inference_ms", *self._class_names])
            self._header_written = True

        row = [
            self._frame_count,
            round(elapsed_seconds, 3),
            round(fps, 3),
            round(inference_ms, 3),
            *[counts.get(name, 0) for name in self._class_names],
        ]
        self._writer.writerow(row)

        self._fps_values.append(fps)
        self._inference_times_ms.append(inference_ms)
        for cls_name, count in counts.items():
            previous = self._max_counts.get(cls_name, 0)
            if count > previous:
                self._max_counts[cls_name] = count

        self._frame_count += 1

    def finalize(self, screenshots_saved: int) -> None:
        self._csv_file.flush()
        self._csv_file.close()

        summary = {
            "run_id": self.run_id,
            "frames_processed": self._frame_count,
            "average_fps": round(mean(self._fps_values), 3) if self._fps_values else 0.0,
            "average_inference_ms": round(mean(self._inference_times_ms), 3)
            if self._inference_times_ms
            else 0.0,
            "max_count_per_class": self._max_counts,
            "screenshots_saved": screenshots_saved,
            "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "frame_csv": str(self.csv_path),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

