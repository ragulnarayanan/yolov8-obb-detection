# Custom Object Detection using YOLOv8 OBB

An end-to-end custom object detection system built with YOLOv8 Oriented Bounding Box (OBB) architecture, trained on 39 custom object classes annotated using Roboflow and executed entirely in Google Colab.

---


## 📌 Overview

This project develops a YOLOv8 OBB model to detect and classify 39 custom objects. Each object class consisted of approximately 100 images, all manually annotated with oriented bounding boxes using Roboflow.
The entire workflow — data annotation, training, evaluation, and inference — was performed in Google Colab, with the dataset, YOLO-format export files, and all project files stored in Google Drive.
Training was manually monitored; once validation accuracy plateaued, training was stopped manually and the best-performing model weights were saved automatically by YOLO.
To validate real-world performance, custom test images were created by collaging 4 different objects into a single image. The model successfully identified each object and drew accurate oriented bounding boxes, confirming strong detection capability on multi-object scenes.

---

## 🏗️ Pipeline

```
Image Collection (~100 images/class)
       ↓
Annotation in Roboflow (OBB format)
       ↓
Dataset Export (YOLO OBB format) → Google Drive
       ↓
YOLOv8s-OBB Training (Google Colab)
       ↓
Manual Early Stopping (on accuracy plateau)
       ↓
Best Weights Saved → Google Drive
       ↓
Inference on Custom Collage Images
```

----

## 📊 Dataset
```

Detail               |    Value
---------------------|-----------------------------------
Total Classes        |  39
Images per Class     | ~100
Annotation Tool      | Roboflow (Oriented Bounding Boxes)
Export Format        | YOLO OBB-compatible
Storage              | Google Drive
Split                | ~80% Train / ~20% Validation

```

Dataset Structure
```
yolo_dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## 🔧 Training Configuration

```
Parameter        |  Value
-----------------|--------------------------------
Model            |  yolov8s-obb.pt (pretrained)
Image Size       |  512 × 512
Batch Size       |  32
Early Stopping   |  Manual (monitored plateau)
Environment      |  Google Colab
GPU              |  NVIDIA A100
```
---

## 📈 Data Augmentation

### Stage 1 – Roboflow (Pre-training)

- Rotation
- Scaling
- Flipping
- Brightness
- Color variation

### Stage 2 – YOLO Built-in (During Training)

- Mosaic augmentation
- MixUp
- HSV color space augmentation
- Horizontal/Vertical flips

----

## 📊 Evaluation Metrics

```
Metric        |   Description
--------------|---------------------------------------
Precision (P) |  Correctness of positive predictions
Recall (R)    |  Ability to detect all true objects
mAP@0.5       |  Detection accuracy at IoU ≥ 0.5
mAP@0.5–0.95  |  Stricter localization accuracy

```
---

## 🚀 Quick Start

### Run Inference
```
pythonmodel = YOLO("runs/obb/train/weights/best.pt")

results = model.predict(
    source="path/to/collage_image.jpg",
    conf=0.25,
    save=True
)
```

### Evaluate

```
pythonmetrics = model.val(data="data.yaml")
print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5-0.95: {metrics.box.map:.4f}")
```
---

## 📂 Output Artifacts

```
yolov8-obb-detection/
├── model/
│   ├── best.pt              # Best model weights
│   └── last.pt              # Last epoch weights
└── assets/
    ├── confusion_matrix.png     # Per-class performance
    └── results.png              # Loss & mAP training curves

```

---

## 🛠️ Tech Stack

```
Component             |  Technology
----------------------|------------------------------
Model                 |  YOLOv8s-OBB (Ultralytics)
Annotation            |  Roboflow
Training Environment  |  Google Colab (A100 GPU)
File Storage          |  Google Drive
Framework             |  Python, Tensorflow
```
---

## ✅ Key Results

- Trained on 39 custom object classes with ~100 images each
- All annotations done using oriented bounding boxes in Roboflow
- Model correctly detected and localized all objects in collage inference tests with multiple objects per frame
- Best weights auto-saved by YOLO upon manual training stop

---

## 📄 License

This project is for educational and research purposes.

---

## 🤝 Acknowledgments

- Ultralytics YOLOv8
- Roboflow
- Google Colab for GPU resources
