import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# --- 1. CONFIGURATION ---
# We try to find the 'lenses' folder automatically
possible_paths = ["lenses", "dataset/lenses", "../lenses"]
dataset_path = None

print(f"Current Working Directory: {os.getcwd()}")

# Look for the folder
for path in possible_paths:
    if os.path.exists(path):
        dataset_path = path
        print(f"✅ Found dataset at: {path}")
        break

if dataset_path is None:
    print("❌ ERROR: Could not find 'lenses' folder.")
    print("ACTION: Make sure your terminal is inside 'ML4Sci-DeepLense-GSoC-Jeevan'")
else:
    # --- 2. LOAD & DISPLAY ---
    # Since you have Model I (Binary), you should have 'no_sub' and 'sub'
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"Found Classes: {classes}")

    plt.figure(figsize=(10, 5))
    
    for i, class_name in enumerate(classes):
        folder_path = os.path.join(dataset_path, class_name)
        all_files = os.listdir(folder_path)
        
        if len(all_files) > 0:
            # Pick a random image
            image_name = random.choice(all_files)
            image_path = os.path.join(folder_path, image_name)
            
            # Show it
            img = mpimg.imread(image_path)
            plt.subplot(1, len(classes), i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"{class_name}\n({image_name})")
            plt.axis('off')

    plt.tight_layout()
    plt.show()