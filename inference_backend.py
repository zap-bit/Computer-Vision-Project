# inference_backend.py
import argparse
import os
import time
from typing import NoReturn

import cv2
from ultralytics import YOLO

from backend_storage import BackendStorage
from inventory_logger import InventoryLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO webcam inference with backend logging.")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Model path (.pt or .engine). Example: runs/detect/shelf_inventory/weights/best.pt",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Write one log row every N frames (default: 1)",
    )
    parser.add_argument(
        "--run-name",
        default="shelf_inventory",
        help="Run name prefix used in backend log folder naming.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional max frames to process before exiting (0 means unlimited).",
    )
    return parser.parse_args()


def _exit_with_error(message: str) -> NoReturn:
    raise SystemExit(message)


def _open_camera(camera_index: int) -> cv2.VideoCapture:
    # On Windows, DirectShow is often more reliable than MSMF for webcams.
    if os.name == "nt":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera_index)


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    cap = _open_camera(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        _exit_with_error("Error: Could not open webcam. Check camera index or permissions.")

    logger = InventoryLogger(run_name=args.run_name)
    storage = BackendStorage(run_id=logger.run_id, base_dir=logger.base_dir)

    print("Press 'q' to quit")
    print("Press 's' to save screenshot")

    fps_history: list[float] = []
    screenshot_count = 0
    frame_index = 0
    session_start = time.time()
    interrupted = False

    try:
        while True:
            ret, frame = cap.read()
            from backend_api import latest_frame
            latest_frame = frame.copy()
            if not ret:
                break

            start_time = time.time()
            results = model.predict(source=frame, conf=args.conf, verbose=False)
            inference_time = (time.time() - start_time) * 1000

            fps = 1000 / inference_time if inference_time > 0 else 0.0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            class_names = model.names
            counts = {name: 0 for name in class_names.values()}

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_idx = int(box.cls[0])
                    conf = float(box.conf[0])

                    cls_name = class_names[cls_idx]
                    counts[cls_name] += 1

                    label = f"{cls_name}: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            y0 = 30
            for cls_name, count in counts.items():
                if count > 0:
                    cv2.putText(
                        frame,
                        f"{cls_name}: {count}",
                        (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    y0 += 30

            cv2.putText(
                frame,
                f"Inference: {inference_time:.1f}ms",
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"FPS: {avg_fps:.1f}",
                (10, y0 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

            if frame_index % max(args.log_interval, 1) == 0:
                elapsed = time.time() - session_start
                logger.log_frame(elapsed, counts, avg_fps, inference_time)
                storage.log_counts(frame_index, elapsed, counts, inference_time, avg_fps)

            cv2.imshow("YOLO Shelf Detector (Backend Logging)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                screenshot_path = f"screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")
                screenshot_count += 1

            frame_index += 1
            if args.max_frames > 0 and frame_index >= args.max_frames:
                print(f"Reached max frames ({args.max_frames}), stopping.")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user. Shutting down cleanly...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.finalize(screenshots_saved=screenshot_count)

    if frame_index == 0:
        _exit_with_error("Error: Camera started but no frames were captured.")

    print("\nSession Stats:")
    if fps_history:
        print(f"Average FPS: {sum(fps_history) / len(fps_history):.1f}")
    print(f"Screenshots saved: {screenshot_count}")
    print(f"Backend logs saved in: {logger.base_dir}")
    latest = storage.get_latest_counts() if frame_index > 0 else {}
    if latest:
        non_zero = {k: v for k, v in latest.items() if v > 0}
        print(f"Latest counts snapshot: {non_zero if non_zero else '{}'}")
    if interrupted:
        print("Run ended early due to interrupt.")
    storage.close()


if __name__ == "__main__":
    main()
