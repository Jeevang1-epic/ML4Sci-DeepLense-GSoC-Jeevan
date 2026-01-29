import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
IMG_SIZE = 64
DATA_PATH = "lenses"

# --- 2. DATA PREPARATION ---
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

print("Loading Test Data...")
# Recreate dataset split to ensure consistency with training
full_dataset = datasets.ImageFolder(root=DATA_PATH)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
# Using fixed seed for reproducibility of the split
_, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

test_dataset.dataset.transform = test_transform 
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. LOAD MODEL ---
print("Loading Best Saved Model...")
model = models.resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 1)

model.load_state_dict(torch.load("best_deeplense_model.pth"))
model = model.to(DEVICE)
model.eval()

# --- 4. EVALUATION ---
print("Calculating ROC Metrics...")
y_true = []
y_scores = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        
        y_true.extend(labels.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

print(f"Final Verified AUC Score: {roc_auc:.4f}")

# --- 5. PLOT RESULTS ---
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='#ff8c00', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('DeepLense Classification Performance (ResNet18)', fontsize=15)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)

output_file = "Final_ROC_Curve.png"
plt.savefig(output_file, dpi=300)
print(f"Graph saved to: {output_file}")