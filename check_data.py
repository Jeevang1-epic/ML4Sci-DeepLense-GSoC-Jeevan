import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# --- 1. CONFIGURATION ---
possible_paths = ["lenses", "dataset/lenses", "../lenses"]
dataset_path = None

print(f"Current Directory: {os.getcwd()}")

# Locate dataset folder
for path in possible_paths:
    if os.path.exists(path):
        dataset_path = path
        print(f"Dataset found at: {path}")
        break

if dataset_path is None:
    print("ERROR: Could not find 'lenses' folder.")
else:
    # --- 2. DISPLAY SAMPLES ---
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"Classes detected: {classes}")

    plt.figure(figsize=(10, 5))
    
    for i, class_name in enumerate(classes):
        folder_path = os.path.join(dataset_path, class_name)
        all_files = os.listdir(folder_path)
        
        if len(all_files) > 0:
            image_name = random.choice(all_files)
            image_path = os.path.join(folder_path, image_name)
            
            img = mpimg.imread(image_path)
            plt.subplot(1, len(classes), i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"{class_name}\n({image_name})")
            plt.axis('off')

    plt.tight_layout()
    plt.show()