# Computer-Vision-Project
Spring 2026 CS 4391 final project

# Automated Retail Inventory Tracking with YOLOv8

Computer Vision course project using NVIDIA Jetson Orin Nano for real-time shelf inventory monitoring with object detection and counting.

##Project Overview

Monitor shelf inventory from camera-based object detection, track class-wise counts, and review performance in one dashboard.

**Hardware:**
- NVIDIA Jetson Orin Nano Developer Kit
- Logitech C270 HD Webcam
- Cam 8MP IMX219 Camera

**Detected Classes:**
- Water bottles
- Oranges
- Apples
- [Add the rest of our custom classes]

---

## Project Structure

```
project/
├── inference.py              # Main detection pipeline
├── train.py                  # Model training script
├── validate.py               # Model evaluation
├── tensorrt_export.py        # TensorRT optimization + benchmarking
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/
│   ├── dataset.yaml          # Dataset configuration
│   ├── images/
│   │   ├── train/           # Training images
│   │   └── val/             # Validation images
│   └── labels/
│       ├── train/           # Training labels
│       └── val/             # Validation labels
└── runs/
    └── detect/
        └── shelf_inventory/
            ├── weights/
            │   ├── best.pt      # Best PyTorch model
            │   └── best.engine  # TensorRT optimized model
            └── [training plots and metrics]
```

