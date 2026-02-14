from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 1.here i amm Loading the YOLOv8 Nano model (pretrained on COCO)[its very fastest, lightest version]
    # The 'n' stands for Nano - optimized for speed and efficiency
    model = YOLO('yolov8n.pt') 

    # 2. Train the model
    # We pass the absolute path to dataset.yaml to avoid path errors
    yaml_path = os.path.abspath("dataset.yaml")
    
    print(f"Starting training on: {yaml_path}")
    
    results = model.train(
        data=yaml_path,
        epochs=100,          # i was not satisfied with 50 Epochs so increased to 100 but 50 is usually enough for this task
        imgsz=64,           # Image size matches our dataset
        batch=16,
        name='deeplense_yolo', # Name of the output folder
        device=0,           # Use GPU (0). Change to 'cpu' if GPU fails.
        pretrained=True,
        plots=True          # Automatically save learning curves
    )

    print("Training Complete. Best model saved in 'runs/detect/deeplense_yolo/weights/'")