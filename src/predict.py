import os
import pandas as pd
from ultralytics import YOLO
import torch

# Load trained model
model = YOLO("models/best.pt").to("cuda" if torch.cuda.is_available() else "cpu")

# Define image directory
image_folder = "dataset/images/test"
image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

# Run inference & save results
results_list = []
for image_file in image_files:
    results = model(os.path.join(image_folder, image_file), device=0)
    for result in results:
        for box in result.boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            results_list.append([image_file, conf, xmin, ymin, xmax, ymax, cls])

# Save predictions
df = pd.DataFrame(results_list, columns=["filename", "confidence", "xmin", "ymin", "xmax", "ymax", "class"])
df.to_csv("results/predictions.csv", index=False)
print("Results saved to results/predictions.csv")
