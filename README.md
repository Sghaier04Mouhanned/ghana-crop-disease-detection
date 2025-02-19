# Ghana Crop Disease Detection  

This project trains a YOLO model to detect crop diseases using object detection. It includes **training, inference, and CSV-to-YOLO format conversion** for further fine-tuning.  

## 🚀 Features  
- YOLO-based crop disease detection  
- CUDA-accelerated inference  
- Outputs predictions as CSV (**filename, confidence, bounding boxes, class**)  
- Converts CSV results into **YOLO-supported annotation format**  

## 📂 Directory Structure  
- `src/train.py` → Train YOLO model  
- `src/predict.py` → Run inference & save CSV  
- `src/csv_to_yolo.py` → Convert predictions CSV to YOLO annotation format  
- `configs/data.yaml` → Dataset paths & class names  
- `models/best.pt` → Trained YOLO model  

## 🔧 Installation  
```bash
pip install -r requirements.txt
