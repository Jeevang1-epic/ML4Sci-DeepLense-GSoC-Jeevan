import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import time
from torch.cuda.amp import autocast, GradScaler # <--- THE SPEED BOOSTER

# --- 1. CONFIGURATION ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print(" TURBO MODE: ENGAGED (Mixed Precision Active)")
else:
    DEVICE = torch.device("cpu")
    print("❌ GPU NOT FOUND. Using CPU.")

BATCH_SIZE = 128  # Increased batch size for speed (AMP uses less memory)
LEARNING_RATE = 0.0001
EPOCHS = 100      # Targeting the moon
IMG_SIZE = 64

# Robust Path Finder
possible_paths = [
    "lenses", 
    "dataset/lenses", 
    os.path.join(os.getcwd(), "lenses"),
    os.path.join(os.getcwd(), "ML4Sci-DeepLense-GSoC-Jeevan", "lenses")
]

DATA_PATH = None
for path in possible_paths:
    if os.path.exists(path):
        DATA_PATH = path
        break
        
if DATA_PATH is None:
    DATA_PATH = r"C:\Users\jeevan kumar\Desktop\ML4Sci-DeepLense-GSoC-Jeevan\lenses"

# --- 2. DATA AUGMENTATION ---
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), # Added Vertical Flip for Space images
    transforms.RandomRotation(30),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

try:
    full_dataset = datasets.ImageFolder(root=DATA_PATH)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_indices, test_indices = random_split(range(len(full_dataset)), [train_size, test_size])
    
    train_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=DATA_PATH, transform=train_transform), train_indices)
    test_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=DATA_PATH, transform=test_transform), test_indices)
    print(f"✅ Loaded {len(full_dataset)} images.")
except Exception as e:
    print("❌ Error loading data.")
    raise e

# Num_workers=0 is safer for Windows Jupyter, but try 2 if you want more speed
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

# --- 3. MODEL: RESNET18 ---
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler() # <--- Initialize Scaler for AMP

# --- 4. TRAINING LOOP ---
print(f"\n--- Starting 100 Epoch Speed Run ---")
best_auc = 0.0
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.float().to(DEVICE).unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Mixed Precision Forward Pass (Faster)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Scaled Backward Pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    
    # Check Validtion every 5 epochs OR if loss is extremely low
    if (epoch + 1) % 5 == 0 or avg_loss < 0.05:
        model.eval()
        y_true, y_scores = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                y_true.extend(labels.cpu().numpy())
                y_scores.extend(probs.cpu().numpy())
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        current_auc = auc(fpr, tpr)
        
        elapsed = (time.time() - start_time) / 60
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | AUC: {current_auc:.4f} | Time: {elapsed:.1f}m")
        
        if current_auc > best_auc:
            best_auc = current_auc
            torch.save(model.state_dict(), "best_deeplense_model.pth")
            print(f"    New Best Saved! ({best_auc:.4f})")

print(f"\n FINAL BEST AUC: {best_auc:.4f}")