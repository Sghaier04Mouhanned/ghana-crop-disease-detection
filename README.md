# Ghana Crop Disease Detection 🌱

This project trains a YOLO model to detect crop diseases using object detection.

## 🚀 Features
- Uses YOLOv10 for detecting 23 crop diseases
- CUDA-accelerated inference
- Outputs predictions as CSV (filename, confidence, bounding boxes, class)

## 📂 Directory Structure
- `src/train.py` → Train YOLO model
- `src/predict.py` → Run inference & save CSV
- `configs/data.yaml` → Dataset paths & class names
- `models/best.pt` → Trained YOLO model

## 🔧 Installation
```bash
pip install -r requirements.txt
