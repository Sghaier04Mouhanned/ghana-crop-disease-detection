from ultralytics import YOLO

# Load model and train
model = YOLO("yolov10x.pt")  
model.train(
    data="configs/data.yaml",  
    epochs=50,  
    imgsz=640,  
    batch=16,  
    device=0  
)
