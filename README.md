# Comparative Analysis of CNN and ScatNet for Image Classification

**Course:** Visual Intelligence  
**Task:** Binary classification of lung cancer histopathological images (Benign vs. Adenocarcinoma)

---

## Overview

This project compares two image classification approaches on the [LC25000 Lung Cancer Histopathological Image Dataset](https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images):

| Model | Approach | Test Accuracy | F1 Score |
|---|---|---|---|
| **ImprovedCNN** | 3 conv blocks, learned filters | **99.00%** | **0.99** |
| **ScatNet** | Fixed Morlet wavelets + MLP | 88.60% | 0.88 |

Both models are trained with **5-fold cross-validation** and analysed using **Explainable AI (XAI)** methods to visualise which image regions and features drive predictions.

---

## Project Structure

```
├── visual-intelligence.ipynb     # Main notebook (all steps)
├── best_kfold_model.pth          # Saved CNN best model weights
├── xai_results/                  # XAI output images
│   ├── patch_permutation_benign_01.png ... _10.png
│   ├── patch_permutation_adenocarcinoma_01.png ... _10.png
│   ├── feature_importance_all.png
│   ├── feature_importance_top20.png
│   ├── feature_importance_by_order.png
│   ├── feature_importance_heatmap.png
│   ├── feature_importance_per_class.png
│   └── importance_summary.txt
└── README.md
```

---

## Dataset

- **Source:** [Kaggle — Lung Cancer Histopathological Images](https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images)
- **Classes:** Benign (0), Adenocarcinoma (1)
- **Total images:** 10,000 (5,000 per class — balanced)
- **Image format:** Grayscale, resized to 64×64 pixels
- **Split:** 8,000 training / 2,000 held-out test

Place the dataset at:
```
/kaggle/input/lung-cancer-histopathological-images/
├── adenocarcinoma/
└── benign/
```

---

## Requirements

Install dependencies:

```bash
pip install torch torchvision
pip install kymatio          # ScatNet wavelet transforms
pip install captum           # XAI feature importance
pip install scikit-learn matplotlib numpy opencv-python Pillow
```

Or install all at once:

```bash
pip install torch torchvision kymatio captum scikit-learn matplotlib numpy opencv-python Pillow
```

> **Note:** The notebook was developed on Kaggle (CPU/MPS). Device is auto-detected (`mps` → `cuda` → `cpu`).

---

## Notebook Structure

The notebook is organized into the following sections:

### 1. Setup & Data Loading
- Install dependencies (`kymatio`, `captum`)
- Import libraries
- Set random seeds (torch seed=42, numpy seed=42)
- Load image paths and labels from dataset directory
- 80/20 train/test split with stratification

### 2. CNN — ImprovedCNN

**Architecture:**

```
Input (1×64×64)
→ Conv2d(1, 16, 3×3) + BatchNorm + ReLU + MaxPool
→ Conv2d(16, 32, 3×3) + BatchNorm + ReLU + MaxPool
→ Conv2d(32, 64, 3×3) + BatchNorm + ReLU + MaxPool
→ Dropout(0.25)
→ Flatten → Linear(4096, 128) → ReLU
→ Linear(128, 64) → ReLU
→ Linear(64, 2)
```

**Training:**
- 5-fold Stratified K-Fold cross-validation
- 10 epochs per fold
- Adam optimiser (lr=0.001, weight_decay=1e-4)
- CrossEntropyLoss, batch size 16
- Best model saved to `best_kfold_model.pth`

**Evaluation:** Confusion matrix + classification report on held-out test set.

### 3. CNN Visualisation
- Conv1 learned filter visualisation (16 filters)
- Feature map outputs for Benign and Adenocarcinoma samples across all 3 conv layers

### 4. ScatNet

**Feature extraction:**
- `Scattering2D(J=2, L=8, shape=(64, 64))` from [Kymatio](https://www.kymat.io/)
- Produces **81 scattering coefficients** per image:
  - Order 0: 1 coefficient (global low-frequency energy)
  - Order 1: 16 coefficients (J × L = 2 × 8, edges at 2 scales and 8 orientations)
  - Order 2: 64 coefficients (texture interactions)
- Each coefficient is a 16×16 spatial map, mean-pooled to a scalar → **81-dim feature vector**

**Classifier (ScatNetClassifier):**

```
Linear(81, 128) → ReLU → Dropout(0.5)
→ Linear(128, 64) → ReLU → Dropout(0.5)
→ Linear(64, 2)
```

**Training:**
- Same 5-fold protocol, 10 epochs, Adam (lr=0.001), batch size 64

**Visualisation:**
- Morlet wavelet filters (J=2 scales × L=8 orientations)
- All 81 scattering coefficient maps (9×9 grid)
- Low-pass scaling filter

### 5. XAI — Explainability

**Part 1: Patch Permutation Heatmaps**
- Divides each 64×64 image into 16×16 patches (4×4 grid)
- For each patch: randomly shuffles pixel values (true permutation, not zeroing) and measures the drop in predicted probability
- Repeated 5 times per patch and averaged
- Applied to both CNN and ScatNet (full pipeline) for **10 images per class**
- Output: side-by-side `Original | CNN heatmap | ScatNet heatmap`
- Saved to `xai_results/patch_permutation_*.png`

**Part 2: Captum Feature Permutation**
- Uses `captum.attr.FeaturePermutation` on the 81-dim scattering feature vector
- Each of the 81 coefficients is shuffled across the batch and output score change is recorded
- Run separately for each class (1,000 samples), then averaged
- Output plots:
  - All 81 coefficients sorted by importance
  - Top 20 most important coefficients
  - Importance grouped by scattering order (0, 1, 2)
  - Heatmap of coefficient importance
  - Per-class comparison (Benign vs. Adenocarcinoma)

---

## Results Summary

### CNN

**Overall test accuracy: 99.00%** (1,978 / 2,000 correct — 983 Benign, 995 Adenocarcinoma)

| Metric | Benign | Adenocarcinoma |
|---|---|---|
| Precision | 0.99 | 0.98 |
| Recall | 0.98 | 0.99 |
| F1-Score | 0.99 | 0.99 |

### ScatNet
| Metric | Value |
|---|---|
| Mean Val Accuracy (5 folds) | 88.31% |
| Mean Val F1 Score (5 folds) | 88.28% |
| Test Accuracy | 88.60% |
| Trainable Parameters | ~22K |

### Key XAI Finding
The ScatNet relies almost entirely on the **zeroth-order scattering coefficient** (global image energy), which has ~10× higher importance than any first-order coefficient. The CNN, by contrast, focuses on **spatially localised regions** of the tissue (glandular structures, nuclear density) corresponding to known discriminative features of adenocarcinoma.

