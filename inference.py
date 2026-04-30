<<<<<<< HEAD
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Use TensorRT if available, fall back to PyTorch
try:
    model = YOLO("segment/runs/detect/shelf_inventory/weights/best.engine")
    print("Loaded TensorRT model")
except:
    model = YOLO("segment/runs/detect/shelf_inventory/weights/best.pt")
    print("Loaded PyTorch model")

CLASSES = ['chocolate', 'granola', 'mouse', 'pen', 'waterbottle']
COLORS = {
    'chocolate':   (0,   140, 255),
    'granola':     (0,   255, 180),
    'mouse':       (255, 100,   0),
    'pen':         (255, 255,   0),
    'waterbottle': (100, 255, 100),
}

=======
# inference.py
import cv2
from ultralytics import YOLO
import time

# MODEL SELECTION
#Option 1: Pretrained (Week 1-2)
# model = YOLO("yolov8n.pt")

#Option 2: Custom PyTorch (Week 3-4)
# model = YOLO("runs/detect/shelf_inventory/weights/best.pt")

#Option 3: TensorRT Optimized (Week 5+)
# model = YOLO("runs/detect/shelf_inventory/weights/best.engine")

#open webcam
>>>>>>> integrate
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

<<<<<<< HEAD
print("Press 'q' to quit | 's' to save screenshot")

=======
print("Press 'q' to quit")
print("Press 's' to save screenshot")

#FPS tracking
>>>>>>> integrate
fps_history = []
screenshot_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
<<<<<<< HEAD

    start = time.time()
    results = model.predict(source=frame, conf=0.35, verbose=False)
    inference_ms = (time.time() - start) * 1000

    fps = 1000 / inference_ms if inference_ms > 0 else 0
    fps_history.append(fps)
    if len(fps_history) > 30:
        fps_history.pop(0)
    avg_fps = sum(fps_history) / len(fps_history)

    counts = {cls: 0 for cls in CLASSES}

    for r in results:
        # Draw segmentation masks
        if r.masks is not None:
            for mask, box in zip(r.masks.xy, r.boxes):
                cls_name = CLASSES[int(box.cls[0])]
                color = COLORS[cls_name]
                pts = np.array(mask, dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                cv2.polylines(frame, [pts], True, color, 2)

        # Count and label boxes
        for box in r.boxes:
            cls_name = CLASSES[int(box.cls[0])]
            conf = float(box.conf[0])
            counts[cls_name] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS[cls_name]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    # ── Count panel (top-left dark background) ──────────────────
    panel_w, panel_h = 220, 30 + len(CLASSES) * 30 + 35
    panel = frame[0:panel_h, 0:panel_w].copy()
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(frame[0:panel_h, 0:panel_w], 0.4,
                    panel, 0.6, 0, frame[0:panel_h, 0:panel_w])

    cv2.putText(frame, "INVENTORY COUNT", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    for i, cls in enumerate(CLASSES):
        y = 22 + (i + 1) * 30
        color = COLORS[cls]
        cv2.putText(frame, f"{cls:<12}: {counts[cls]}",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2)

    # FPS + inference time
    cv2.putText(frame,
                f"FPS: {avg_fps:.1f}  |  {inference_ms:.1f}ms",
                (8, panel_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("StockSync — Shelf Inventory", frame)

=======
    
    #measure inference time
    start_time = time.time()
    results = model.predict(source=frame, conf=0.3, verbose=False)
    inference_time = (time.time() - start_time) * 1000  # ms
    
    #calculate FPS
    fps = 1000 / inference_time if inference_time > 0 else 0
    fps_history.append(fps)
    if len(fps_history) > 30:  # Keep last 30 frames
        fps_history.pop(0)
    avg_fps = sum(fps_history) / len(fps_history)
    
    # get class names and counts
    class_names = model.names
    counts = {name: 0 for name in class_names.values()}
    
    #draw boxes and count
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])
            
            cls_name = class_names[cls_idx]
            counts[cls_name] += 1
            
            label = f"{cls_name}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    #display counts
    y0 = 30
    for cls_name, cnt in counts.items():
        if cnt > 0:
            cv2.putText(frame, f"{cls_name}: {cnt}", (10, y0),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            y0 += 30
    
    #display performance metrics
    cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, y0),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, y0+25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    
    #show frame
    cv2.imshow("YOLO Shelf Detector", frame)
    
>>>>>>> integrate
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
<<<<<<< HEAD
        path = f"screenshot_{screenshot_count:03d}.jpg"
        cv2.imwrite(path, frame)
        print(f"Saved: {path}")
=======
        # Save screenshot
        screenshot_path = f"screenshot_{screenshot_count:03d}.jpg"
        cv2.imwrite(screenshot_path, frame)
        print(f"Screenshot saved: {screenshot_path}")
>>>>>>> integrate
        screenshot_count += 1

cap.release()
cv2.destroyAllWindows()
<<<<<<< HEAD
print(f"\nAvg FPS: {sum(fps_history)/len(fps_history):.1f}")
=======

print(f"\nSession Stats:")
print(f"Average FPS: {sum(fps_history)/len(fps_history):.1f}")
>>>>>>> integrate
print(f"Screenshots saved: {screenshot_count}")
