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
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Press 'q' to quit")
print("Press 's' to save screenshot")

#FPS tracking
fps_history = []
screenshot_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
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
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save screenshot
        screenshot_path = f"screenshot_{screenshot_count:03d}.jpg"
        cv2.imwrite(screenshot_path, frame)
        print(f"Screenshot saved: {screenshot_path}")
        screenshot_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"\nSession Stats:")
print(f"Average FPS: {sum(fps_history)/len(fps_history):.1f}")
print(f"Screenshots saved: {screenshot_count}")
