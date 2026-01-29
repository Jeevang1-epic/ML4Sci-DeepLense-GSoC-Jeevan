# 🌌 DeepLense: Gravitational Lensing Classification
### GSoC 2026 Common Task | ML4Sci Foundation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![AUC Score](https://img.shields.io/badge/AUC-1.0000-FFD700?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📖 Project Overview
This repository contains a high-performance deep learning solution for the **ML4Sci DeepLense Common Task (Model I)**. The objective is to classify strong gravitational lensing images into two categories:
1.  **No Substructure (`no_sub`):** Smooth dark matter halos.
2.  **Substructure (`sub`):** Halos containing dark matter sub-halos.

The solution implements a **ResNet18** architecture optimized for single-channel scientific data, utilizing **Mixed Precision (AMP)** training to achieve a verified **ROC AUC score of 1.0000**.

---

## 🔬 Methodology

### 1. Data Visualization
The dataset consists of single-channel (grayscale) simulated lensing images. Below is a sample visualization showing the distinct difference between the two classes:

![Data Visualization](data_visualization.png)
*Left: A smooth halo (no_sub). Right: A halo with substructure (sub).*

### 2. Model Architecture
* **Backbone:** ResNet18 (Pre-trained), adapted for 1-channel input.
* **Optimization:** Adam Optimizer with Binary Cross Entropy (`BCEWithLogitsLoss`).
* **Performance Tuning:**
    * **Mixed Precision Training (AMP):** Utilized `torch.cuda.amp` for 2x training speed on Tensor Core GPUs.
    * **Augmentation Strategy:** Random Rotations (30°), Horizontal Flips, and Vertical Flips to enforce rotational invariance of space imagery.

---

## 📊 Results & Evaluation

The model was trained for **100 epochs**, achieving perfect separation on the test set.

| Metric | Score |
| :--- | :--- |
| **AUC Score** | **1.0000** |
| **Test Accuracy** | **100%** |
| **Convergence** | < 20 Epochs |

### ROC Curve Analysis
The Receiver Operating Characteristic (ROC) curve below demonstrates the model's perfect True Positive Rate vs. False Positive Rate performance (Area Under Curve = 1.0).

![ROC Curve](Final_ROC_Curve.png)

---

## 🛠️ Installation & Usage

### 1. Setup Environment
To set up the project, clone the repository and install the required dependencies (Python 3.10+ and PyTorch required):

    # Clone the repository
    git clone [https://github.com/YOUR_USERNAME/ML4Sci-DeepLense-GSoC-Jeevan.git](https://github.com/YOUR_USERNAME/ML4Sci-DeepLense-GSoC-Jeevan.git)
    cd ML4Sci-DeepLense-GSoC-Jeevan

    # Install dependencies
    pip install torch torchvision matplotlib scikit-learn numpy

### 2. Training
To reproduce the training results using the GPU (if available):

    python train_model.py

### 3. Evaluation
To load the saved weights (`best_deeplense_model.pth`) and generate the ROC curve:

    python evaluate_model.py

### 4. Data Inspection
To inspect random samples from the dataset to understand the problem:

    python check_data.py

## 📂 Repository Structure

    ├── lenses/                  # Dataset directory
    ├── train_model.py           # Training pipeline (ResNet18 + AMP)
    ├── evaluate_model.py        # Inference & Visualization script
    ├── check_data.py            # Data exploration utility
    ├── best_deeplense_model.pth # Saved Model Weights (State Dict)
    ├── Final_ROC_Curve.png      # Performance Graph
    ├── data_visualization.png   # Data Sample Image
    └── README.md                # Project Documentation
