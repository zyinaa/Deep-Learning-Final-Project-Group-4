<p align="center">
  <img src="Code/app/icon.png" alt="DermAI Logo" width="120"/>
</p>

<h1 align="center">DermAI</h1>
<p align="center">
  <b>Skin Cancer Risk Classification System</b><br/>
  <sub>Powered by Vision Transformer (ViT-base-patch16-224)</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-ViT--base--patch16--224-00B4D8?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset-HAM10000+ISIC2020-1D9E75?style=flat-square"/>
  <img src="https://img.shields.io/badge/Classes-10-EF9F27?style=flat-square"/>
  <img src="https://img.shields.io/badge/Test_F1-0.6678-E24B4A?style=flat-square"/>
  <img src="https://img.shields.io/badge/HR_Recall-81%25_(clinical)-FF2D2D?style=flat-square"/>
  <img src="https://img.shields.io/badge/GWU-DATS_6303-0D1F3C?style=flat-square"/>
</p>

---

## Overview

**DermAI** is a skin cancer risk classification system that uses a fine-tuned **Vision Transformer (ViT-base-patch16-224)** to classify 10 skin conditions and compute a **Cancer Risk Score (0–100)**. The system provides explainable predictions through attention map visualization, and includes a **Clinical Mode** with ROC-optimized thresholds to maximize recall for high-risk conditions.

### Key Features
- **10-class** skin condition classification
- **Cancer Risk Score** (0–100) weighted by clinical malignancy severity
- **Clinical Mode** with ROC-optimized thresholds for high-risk classes
- **Attention map** visualization for model explainability
- **4 Risk Groups**: High / Elevated / Moderate / Low
- **PDF report** generation with charts and analysis
- **Model comparison** between ViT-base and EfficientNet-B3
- **Real-time inference** via Streamlit web application

---

## Demo

> Live demo running on AWS:

```
http://34.207.138.130:8501
```

> HuggingFace Space (VIT, EfficientNet, CPU):

```
https://huggingface.co/spaces/zyinaa/DermAI
```

---

## Results

| Model | Mode | Test F1 | Accuracy | High Risk Recall |
|---|---|---|---|---|
| **ViT-base-patch16-224** | Standard | **0.6678** | **76%** | 68% |
| **ViT-base-patch16-224** | Clinical | 0.6338 | 70% | **81%** |
| EfficientNet-B3 | Standard | 0.6488 | 74% | 58% |
| EfficientNet-B3 | Clinical | 0.6305 | 71% | 76% |

> **High Risk Recall** = recall across Melanoma + BCC + SCC combined (204 test samples)

> **Clinical Mode** uses ROC-optimized thresholds to prioritize recall for high-risk classes at the cost of overall accuracy — an acceptable trade-off in clinical settings where missing a melanoma is far more dangerous than a false alarm.

### Per-Class Performance (ViT Standard)

| Class | Risk Group | F1 | Recall |
|---|---|---|---|
| Healthy Skin | Low | 1.00 | 1.00 |
| Squamous Cell Carcinoma | High | 0.90 | 0.81 |
| Tinea/Ringworm | Low | 0.80 | 0.83 |
| Vascular Lesion | Low | 0.82 | 0.78 |
| Melanocytic Nevi | Moderate | 0.84 | 0.78 |
| Dermatofibroma | Low | 0.69 | 0.90 |
| Benign Keratosis | Moderate | 0.49 | 0.48 |
| Basal Cell Carcinoma | High | 0.43 | 0.86 |
| Melanoma | High | 0.37 | 0.40 |
| Actinic Keratosis | Elevated | 0.34 | 0.25 |

---

## Dataset

### HAM10000 (Primary)
- 10,015 dermoscopic images across 7 classes
- Source: [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) / [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

### Supplementary Kaggle Data
- Tinea/Ringworm: 122 images
- Squamous Cell Carcinoma: 322 images
- Healthy Skin (normal + dry + oily): 2,756 images

### ISIC 2020 (Melanoma Augmentation)
- 584 additional melanoma images to address class imbalance
- Split: Train 467 / Val 117
- Source: [Kaggle - ISIC 2020 224×224](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-224x224-resized)

### Final Dataset
| Split | Images |
|---|---|
| Train | 10,357 |
| Validation | 2,096 |
| Test | 1,346 |
| **Total** | **13,799** |

---

## 10 Classes and Cancer Risk Weights

| Class | Code | Source | Risk Weight | Risk Group |
|---|---|---|---|---|
| Melanoma | mel | HAM10000 + ISIC2020 | 1.00 | 🔴 High |
| Squamous Cell Carcinoma | scc | Kaggle | 0.90 | 🔴 High |
| Basal Cell Carcinoma | bcc | HAM10000 | 0.80 | 🔴 High |
| Actinic Keratosis | akiec | HAM10000 | 0.65 | 🟠 Elevated |
| Benign Keratosis | bkl | HAM10000 | 0.10 | 🟡 Moderate |
| Melanocytic Nevi | nv | HAM10000 | 0.10 | 🟡 Moderate |
| Dermatofibroma | df | HAM10000 | 0.05 | 🟢 Low |
| Vascular Lesion | vasc | HAM10000 | 0.05 | 🟢 Low |
| Tinea/Ringworm | tinea | Kaggle | 0.00 | 🟢 Low |
| Healthy Skin | normal | Kaggle | 0.00 | 🟢 Low |

**Cancer Risk Score** = Σ(P(class_i) × weight_i) × 100

---

## Clinical Mode

Clinical Mode uses ROC-optimized thresholds (Youden's J) for high-risk classes:

| Class | ViT Threshold | ENet Threshold | ViT TPR | ViT AUC |
|---|---|---|---|---|
| Melanoma | 0.117 | 0.152 | 80% | 0.793 |
| Basal Cell Carcinoma | 0.131 | 0.050 | 98% | 0.967 |
| Squamous Cell Carcinoma | 0.074 | 0.078 | 91% | 0.979 |
| Actinic Keratosis | 0.059 | 0.121 | 81% | 0.906 |

---

## Model Architecture

### Vision Transformer (ViT-base-patch16-224)
- Pre-trained on ImageNet-21k (14M images, 21k classes)
- Input: 224×224 RGB images
- Patch size: 16×16 (196 patches + 1 CLS token)
- Encoder: 12 transformer layers, 12 attention heads
- Hidden dimension: 768
- Parameters: 86M

### Two-Phase Fine-tuning Strategy
```
Phase 1 (5 epochs):
  Freeze ViT encoder → Train classification head only
  Learning rate: 1e-3

Phase 2 (25 epochs):
  Unfreeze all layers → Full fine-tuning
  Learning rate: 5e-5
  CosineAnnealingLR + Label smoothing 0.1
```

### Risk-Weighted Loss
```python
risk_multipliers = {
    mel:   8.0x,   # Highest risk, worst recall
    bcc:   3.0x,
    akiec: 3.0x,
    scc:   2.5x,
    df:    3.0x,   # Very few samples
}
criterion = nn.CrossEntropyLoss(weight=class_weights * risk_multipliers)
```

---

## Repository Structure

```
Deep-Learning-Final-Project-Group-4/
├── icon.png
├── README.md
├── Group-Proposal/
├── Final-Group-Project-Report/
├── Final-Group-Presentation/
└── Code/
    ├── preprocessing/
    │   └── dataset.py              # Dataset class, transforms, ISIC2020 loading
    ├── training/
    │   ├── train_vit.py            # ViT fine-tuning (Phase 1 + Phase 2)
    │   └── train_efficientnet.py   # EfficientNet-B3 baseline
    ├── models/
    │   └── saved/                  # Model weights (not on GitHub)
    ├── evaluation/
    │   ├── evaluate.py             # Confusion matrix, per-class F1
    │   └── find_threshold.py       # ROC analysis, optimal thresholds
    ├── app/
    │   ├── app.py                  # DermAI Streamlit application
    │   └── icon.png
    └── data/
        └── raw/
            ├── ham10000/
            ├── kaggle_diseases/
            └── isic2020/
```

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/zyinaa/Deep-Learning-Final-Project-Group-4.git
cd Deep-Learning-Final-Project-Group-4
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Download Data
```bash
# HAM10000
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p Code/data/raw/ham10000 --unzip

# Kaggle skin diseases
kaggle datasets download -d haroonalam16/20-skin-diseases-dataset -p Code/data/raw/kaggle_diseases --unzip
kaggle datasets download -d shakyadissanayake/oily-dry-and-normal-skin-types-dataset -p Code/data/raw/kaggle_diseases --unzip

# ISIC 2020 melanoma
kaggle datasets download -d nischaydnk/isic-2020-jpg-224x224-resized -p Code/data/raw/isic2020 --unzip
```

### 4. Train Models
```bash
cd Code
python3 training/train_efficientnet.py   # Baseline
python3 training/train_vit.py            # Main model
```

### 5. Run DermAI
```bash
streamlit run Code/app/app.py --server.port 8501 --server.address 0.0.0.0
```

---

## DermAI App Features

### Cancer Risk Score
| Score | Level | Color |
|---|---|---|
| 0–20 | Low Risk | 🟢 Green |
| 20–50 | Moderate Risk | 🟡 Yellow |
| 50–75 | Elevated Risk | 🟠 Orange |
| 75–100 | High Risk | 🔴 Red |

### Attention Map
ViT's self-attention weights overlaid on the input image — analogous to the ABCDE criteria used by dermatologists.

### Clinical Mode
ROC-optimized thresholds boosting recall for high-risk classes. Toggle in the sidebar.

### PDF Report
Downloadable report with risk summary, class probability chart, and model metadata.

### Model Comparison
Side-by-side ViT-base vs EfficientNet-B3 with donut charts for both models.

---

## AWS Training Environment

```
Instance:  g5.xlarge
GPU:       NVIDIA A10G (23.6 GB VRAM)
CUDA:      11.8
PyTorch:   2.7.1+cu118
Python:    3.12.3
```

---

## Citation

```bibtex
@article{tschandl2018ham10000,
  title={The HAM10000 dataset, a large collection of multi-source
         dermatoscopic images of common pigmented skin lesions},
  author={Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal={Scientific data},
  volume={5},
  pages={180161},
  year={2018}
}

@article{dosovitskiy2020vit,
  title={An image is worth 16x16 words: Transformers for image
         recognition at scale},
  author={Dosovitskiy, Alexey and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

---

## Team

| Member | Responsibilities |
|---|---|
| Enoch Yin | Dataset pipeline, ViT fine-tuning, EfficientNet baseline, cancer risk score, attention maps, clinical mode, model evaluation, deployment |
| Gary Liang | EDA, data augmentation, Streamlit app, demo preparation, report |

---

## License

Educational use only — GWU DATS 6303 Deep Learning | Spring 2026  
HAM10000 dataset licensed under CC-BY-NC-SA-4.0

---

<p align="center">
  <sub>Built with ViT · HAM10000 · ISIC2020 · Streamlit · PyTorch</sub>
</p>
