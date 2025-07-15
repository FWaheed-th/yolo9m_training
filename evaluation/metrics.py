import torch
from ultralytics import YOLO
from pynvml import *

# Load model
model = YOLO('runs/detect/train4/weights/best.pt')

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# GPU monitoring
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
gpu_util = nvmlDeviceGetUtilizationRates(handle).gpu
gpu_mem = nvmlDeviceGetMemoryInfo(handle).used / 1024**2
print(f"GPU Utilization: {gpu_util}%")
print(f"GPU Memory Used: {gpu_mem:.2f} MB")

# Run validation
metrics = model.val(data='data/data.yaml', imgsz=640, batch=16, device='cuda:0')

# Print results
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

# Save to file
with open('evaluation/yolo_metrics_train2.txt', 'w') as f:
    f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
    f.write(f"Precision: {metrics.box.mp:.4f}\n")
    f.write(f"Recall: {metrics.box.mr:.4f}\n")
    f.write(f"GPU Utilization: {gpu_util}%")
    f.write(f"GPU Memory Used: {gpu_mem:.2f} MB")
