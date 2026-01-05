# 🚧 Road Damage Detection using YOLOv8

## 📌 Overview
Safe and well-maintained road infrastructure is essential for efficient transportation and economic growth. Manual inspection of road surfaces for defects such as cracks and potholes is time-consuming, costly, and subjective.

This project presents an AI-powered computer vision solution that automatically detects and classifies various types of road damage from images using a YOLOv8 object detection model.

---

## 🎯 Problem Statement
Given an image of a road surface, the task is to:
- Detect all instances of road damage
- Classify each instance into one of five predefined damage categories
- Output bounding boxes with confidence scores in YOLO format

The solution is evaluated using Mean Average Precision (mAP) on a hidden test set.

---

## 🗂 Dataset
**Road Damage Detection 2022 (RDD2022)**  
A large-scale, multi-national dataset containing over 47,000 high-resolution road images.

### Data Split
- Training Set – images with labels  
- Validation Set – images with labels  
- Test Set – images without labels  

### Damage Classes
| Class ID | Damage Type |
|--------|------------|
| 0 | Longitudinal Crack |
| 1 | Transverse Crack |
| 2 | Alligator Crack |
| 3 | Other Corruption |
| 4 | Pothole |

---

## 🧠 Model Architecture
- Model: YOLOv8m
- Framework: Ultralytics YOLOv8
- Pretrained Weights: COCO-pretrained
- Input Size: 640 × 640

YOLOv8 was chosen for its high accuracy, fast inference speed, and strong object localization capability.

---

## ⚙️ Training Details

### Training Environment
Training was performed on Kaggle using an NVIDIA Tesla T4 GPU due to dataset size and computational requirements.

### Training Command
```bash
yolo detect train \
  model=yolov8m.pt \
  data=rdd.yaml \
  epochs=30 \
  imgsz=640 \
  batch=16

## 🔧 Data Augmentation & Optimization

### Data Augmentation
To improve generalization and robustness, the following augmentation techniques were applied during training:

- Mosaic augmentation  
- HSV color augmentation  
- Horizontal flipping  
- Random scaling and translation  

### Optimization Strategy
- Optimizer: **Auto** (Adam/SGD selected internally by YOLO)
- Early stopping enabled to prevent overfitting
- COCO-pretrained weights used for faster convergence

---

## 📊 Results

After **30 epochs** of training, the model achieved the following performance on the validation set:

| Metric | Value |
|------|------|
| mAP@0.5 | **0.62** |
| mAP@0.5:0.95 | **0.33** |

### Class-wise Performance
- Best performance observed for **Other Corruption** and **Alligator Crack**
- **Pothole detection** remains challenging due to high size variation and occlusion

---

## 🔍 Inference & Prediction

Predictions were generated on the test set using the trained model (`best.pt`).

Each test image produces a corresponding `.txt` file in YOLO format:

<class_id> <x_center> <y_center> <width> <height> <confidence_score>

Inference logic is implemented in `infer.py`.

---

## 📦 Submission Format

The final submission is structured as follows:

submission.zip
└── predictions/
    ├── 000001.txt
    ├── 000002.txt
    └── ...

- One prediction file per test image
- Empty files included where no damage is detected
- Fully compliant with competition requirements

---

## 📁 Project Structure

road-damage-detection/
│
├── README.md
├── rdd.yaml
├── infer.py
├── requirements.txt
├── best.pt
├── kaggle/
│   └── training_notebook.ipynb
└── submission.zip

---

## 📜 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt

---

## ✅ Conclusion

This project demonstrates a robust and scalable approach to automated road damage detection using deep learning. By leveraging YOLOv8 and GPU acceleration, the system achieves strong detection performance and produces competition-compliant predictions suitable for real-world deployment and evaluation.

---

##📎 References
- Ultralytics YOLOv8: https://docs.ultralytics.com
- RDD2022 Dataset

---