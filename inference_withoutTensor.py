#inference.py -webcam detection pipeline
import cv2
from ultralytics import YOLO

#load model GAP:swap this line when custom model is ready
model = YOLO("yolov8n.pt")  #using pretrained for now
#model = YOLO("runs/detect/shelf_inventory/weights/best.pt")  our custom model to be inserted

#open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #run YOLO inference
    results = model.predict(source=frame, conf=0.3, verbose=False)
    
    #get class names from model
    class_names = model.names
    counts = {name: 0 for name in class_names.values()}
    
    #draw boxes and count
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 =map(int, box.xyxy[0])
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])
            
            cls_name = class_names[cls_idx]  #GAP:use actual class name
            counts[cls_name] += 1
            
            label = f"{cls_name}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    #display counts
    y0 = 30
    for cls_name, cnt in counts.items():
        if cnt > 0:  # Only show classes that are detected
            cv2.putText(frame, f"{cls_name}: {cnt}", (10, y0),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            y0 += 30
    
    #show frame
    cv2.imshow("YOLO Multi-Class Pipeline", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
