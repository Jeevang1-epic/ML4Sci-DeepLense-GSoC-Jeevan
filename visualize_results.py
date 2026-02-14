from ultralytics import YOLO
import os
import random
import matplotlib.pyplot as plt
import cv2

# --- CONFIGURATION ---
MODEL_PATH = "runs/detect/deeplense_yolo2/weights/best.pt" 
TEST_IMAGES_DIR = "yolo_dataset/images/val"
OUTPUT_IMAGE = "YOLO_Predictions.png"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Could not find model at {MODEL_PATH}")
    print("Check your 'runs/detect' folder and update the path in the script!")
    exit()

model = YOLO(MODEL_PATH)

image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith('.jpg')]
selected_files = random.sample(image_files, 6)


plt.figure(figsize=(12, 8))
print(f"Running inference on {len(selected_files)} images...")

for i, file_name in enumerate(selected_files):
    img_path = os.path.join(TEST_IMAGES_DIR, file_name)
    
    # Run Prediction
    results = model.predict(img_path, conf=0.25, verbose=False)
    
    # Plot the result
    result_img = results[0].plot(labels=False) 
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB) # Fix colors for matplotlib
    
    plt.subplot(2, 3, i + 1)
    plt.imshow(result_img)
    plt.axis('off')
    plt.title(f"Detection {i+1}", fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300)
print(f"Success! Prediction grid saved as: {OUTPUT_IMAGE}")