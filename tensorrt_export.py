# tensorrt_export.py - Convert model to TensorRT for Jetson optimization
from ultralytics import YOLO
import cv2
import time
import numpy as np

def export_to_tensorrt(model_path='runs/detect/shelf_inventory/weights/best.pt'):
    """
    Convert PyTorch model to TensorRT engine
    """
    print("  Loading PyTorch model...")
    model = YOLO(model_path)
    
    print("  Exporting to TensorRT (this may take 5-10 minutes)...")
    #export to TensorRT format
    model.export(
        format='engine',        # TensorRT engine
        imgsz=640,              # Input image size
        half=True,              # Use FP16 precision (faster on Jetson)
        device=0,               # GPU device
        workspace=4,            # Max workspace size in GB
        simplify=True,          # Simplify ONNX model
        verbose=True
    )
    
    #tensorRT model path
    engine_path = model_path.replace('.pt', '.engine')
    print(f"\n  TensorRT model saved to: {engine_path}")
    
    return engine_path


def benchmark_models(pytorch_path, tensorrt_path, num_frames=100):
    """
    Compare PyTorch vs TensorRT inference speed
    """
    print("\n  Benchmarking Models...")
    print("=" * 50)
    
    #load both models
    print("Loading PyTorch model...")
    pytorch_model = YOLO(pytorch_path)
    
    print("Loading TensorRT model...")
    tensorrt_model = YOLO(tensorrt_path)
    
    #open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  Webcam not available, using dummy images")
        # Create dummy test images
        test_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                      for _ in range(num_frames)]
    else:
        # Capture real frames
        test_frames = []
        print(f"Capturing {num_frames} test frames...")
        for i in range(num_frames):
            ret, frame = cap.read()
            if ret:
                test_frames.append(frame)
        cap.release()
    
    #benchmark PyTorch
    print(f"\n  Testing PyTorch model ({len(test_frames)} frames)...")
    pytorch_times = []
    for frame in test_frames:
        start = time.time()
        _ = pytorch_model.predict(source=frame, conf=0.3, verbose=False)
        pytorch_times.append((time.time() - start) * 1000)  # Convert to ms
    
    # benchmark TensorRT
    print(f"⚡ Testing TensorRT model ({len(test_frames)} frames)...")
    tensorrt_times = []
    for frame in test_frames:
        start = time.time()
        _ = tensorrt_model.predict(source=frame, conf=0.3, verbose=False)
        tensorrt_times.append((time.time() - start) * 1000)  # Convert to ms
    
    #calculate statistics
    pytorch_avg = np.mean(pytorch_times)
    pytorch_std = np.std(pytorch_times)
    tensorrt_avg = np.mean(tensorrt_times)
    tensorrt_std = np.std(tensorrt_times)
    
    speedup = pytorch_avg / tensorrt_avg
    
    # Print results
    print("\n" + "=" * 50)
    print("--BENCHMARK RESULTS--")
    print("=" * 50)
    print(f"\nPyTorch Model:")
    print(f"  Average inference time: {pytorch_avg:.2f} ms (±{pytorch_std:.2f})")
    print(f"  FPS: {1000/pytorch_avg:.1f}")
    
    print(f"\nTensorRT Model:")
    print(f"  Average inference time: {tensorrt_avg:.2f} ms (±{tensorrt_std:.2f})")
    print(f"  FPS: {1000/tensorrt_avg:.1f}")
    
    print(f"\n  Speedup: {speedup:.2f}x faster")
    print(f"   Time saved per frame: {pytorch_avg - tensorrt_avg:.2f} ms")
    print("=" * 50)
    
    return {
        'pytorch': {'avg': pytorch_avg, 'std': pytorch_std, 'fps': 1000/pytorch_avg},
        'tensorrt': {'avg': tensorrt_avg, 'std': tensorrt_std, 'fps': 1000/tensorrt_avg},
        'speedup': speedup
    }


def live_comparison():
    """
    Side-by-side live comparison (optional demo)
    """
    print("\n  Live Comparison Mode")
    print("Running both models on live webcam...")
    print("Press 'q' to quit")
    
    pytorch_model = YOLO('runs/detect/shelf_inventory/weights/best.pt')
    tensorrt_model = YOLO('runs/detect/shelf_inventory/weights/best.engine')
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #pyTorch inference
        start = time.time()
        pytorch_results = pytorch_model.predict(source=frame, conf=0.3, verbose=False)
        pytorch_time = (time.time() - start) * 1000
        
        #tensorRT inference
        start = time.time()
        tensorrt_results = tensorrt_model.predict(source=frame, conf=0.3, verbose=False)
        tensorrt_time = (time.time() - start) * 1000
        
        #display info
        cv2.putText(frame, f"PyTorch: {pytorch_time:.1f}ms", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"TensorRT: {tensorrt_time:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Speedup: {pytorch_time/tensorrt_time:.2f}x", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        
        cv2.imshow("Live Comparison", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Step 1: Export to TensorRT
    print("STEP 1: Export to TensorRT")
    print("-" * 50)
    
    pytorch_model_path = 'runs/detect/shelf_inventory/weights/best.pt'
    tensorrt_model_path = export_to_tensorrt(pytorch_model_path)
    
    #step 2: Benchmark
    print("\nSTEP 2: Benchmark Performance")
    print("-" * 50)
    
    results = benchmark_models(
        pytorch_path=pytorch_model_path,
        tensorrt_path=tensorrt_model_path,
        num_frames=100  # Test on 100 frames
    )
    
    #optional: Live comparison
    print("\n💡 Want to see live comparison? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        live_comparison()
    
    print("\n  TensorRT optimization complete!")
    print(f"  Use this path in inference.py: {tensorrt_model_path}")
