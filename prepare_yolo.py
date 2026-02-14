import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
SOURCE_DATA_PATH = "lenses/sub"
OUTPUT_DIR = "yolo_dataset"
IMG_SIZE = 64

# Initialize directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

print(f"Scanning {SOURCE_DATA_PATH}...")

try:
    images = [f for f in os.listdir(SOURCE_DATA_PATH) if f.endswith('.jpg')]
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
except FileNotFoundError:
    print(f"Error: Directory '{SOURCE_DATA_PATH}' not found.")
    exit()

def process_batch(image_list, split_name):
    print(f"Processing {split_name} set...")
    
    for img_name in tqdm(image_list):
        img_path = os.path.join(SOURCE_DATA_PATH, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Thresholding to isolate lens features
        _, thresh = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Select largest contour as the gravitational lens
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # Normalize coordinates for YOLO format
            x_center = (x + w / 2) / IMG_SIZE
            y_center = (y + h / 2) / IMG_SIZE
            width = w / IMG_SIZE
            height = h / IMG_SIZE
            
            # Copy image to dataset folder
            shutil.copy(img_path, os.path.join(OUTPUT_DIR, 'images', split_name, img_name))
            
            # Generate label file
            label_name = img_name.replace(".jpg", ".txt")
            label_path = os.path.join(OUTPUT_DIR, 'labels', split_name, label_name)
            
            with open(label_path, "w") as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

process_batch(train_imgs, 'train')
process_batch(val_imgs, 'val')

# Generate dataset configuration
yaml_content = f"""
path: ../{OUTPUT_DIR} 
train: images/train
val: images/val

nc: 1
names: ['gravitational_lens']
"""

with open("dataset.yaml", "w") as f:
    f.write(yaml_content)

print("Dataset generation complete.")
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")