#train.py-Train custom YOLO model
from ultralytics import YOLO

def train_model():
    #load pretrained weights as starting point
    model = YOLO('yolov8n.pt')
    
    #train on our dataset
    results = model.train(
        data='data/dataset.yaml',  #GAP: will change after data is provided
        epochs=100,
        imgsz=640,   #512 or 416 if there are memory issues
        batch=8,    #GAP: adjust based on Jetson memory if we get CUDA outof memory error we can use 4 or 2 or even 1 
        device=0,    #GAP: Use GPU
        project='runs/detect',
        name='shelf_inventory',
        patience=15,    #early stopping if no improvement
        save=True,
        plots=True,    #save training curves
        val=True    #run validation during training
    )
    
    print("\n---Training complete---")
    print(f"Best model saved to: runs/detect/shelf_inventory/weights/best.pt")
    
    return results

if __name__ == "__main__":
    train_model()
