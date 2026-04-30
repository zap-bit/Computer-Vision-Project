# validate.py - Test model accuracy
from ultralytics import YOLO

def validate_model():
    model = YOLO('/content/runs/segment/runs/detect/shelf_inventory/weights/best.pt')

    metrics = model.val(
        data='/content/DataLabelling/data.yaml',
        imgsz=640,
        batch=8,
        plots=True,  #generates confusion matrix, PR curves
        save_json=True
    )
    
    print(f"\n  Validation Results:")
    print(f"Box  mAP@50:    {metrics.box.map50:.3f}")
    print(f"Box  mAP@50-95: {metrics.box.map:.3f}")
    print(f"Mask mAP@50:    {metrics.seg.map50:.3f}")
    print(f"Mask mAP@50-95: {metrics.seg.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")
    
    return metrics

if __name__ == "__main__":
    validate_model()
