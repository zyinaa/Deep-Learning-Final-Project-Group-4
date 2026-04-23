# Skin Cancer Risk Classification Using Vision Transformer (ViT)

**Deep Learning Final Project — Group 4**  
**Course:** DATS 6303 Deep Learning | George Washington University  
**Instructor:** Prof. Amir Jafari  

---

## Project Overview

A multi-class skin condition classifier that uses a fine-tuned **Vision Transformer (ViT-base-patch16-224)** to classify 10 skin conditions and compute a **Cancer Risk Score (0–100)**. The system provides explainable predictions through attention map visualization, showing exactly which regions of the image drove the model's decision.

### Key Features
- 10-class skin condition classification
- Cancer Risk Score weighted by clinical malignancy severity
- Attention map visualization for model explainability
- Real-time inference via Streamlit web application
- Live camera capture support for mobile use

---

## Results

| Model | Test Macro F1 | Accuracy |
|---|---|---|
| **ViT-base-patch16-224** | **0.7652** | **86%** |
| EfficientNet-B3 (baseline) | 0.7440 | 83% |

### Per-Class Performance (ViT)

| Class | F1 Score |
|---|---|
| Healthy Skin | 1.00 |
| Squamous Cell Carcinoma | 0.95 |
| Vascular Lesion | 0.90 |
| Melanocytic Nevi | 0.91 |
| Tinea/Ringworm | 0.87 |
| Basal Cell Carcinoma | 0.73 |
| Benign Keratosis | 0.68 |
| Melanoma | 0.54 |
| Actinic Keratosis | 0.57 |
| Dermatofibroma | 0.50 |

---

## Dataset

### HAM10000 (Primary)
- 10,015 dermoscopic images across 7 classes
- Source: [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) / [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- Citation: Tschandl et al., Scientific Data 2018

### Supplementary Kaggle Data
- Tinea/Ringworm: 122 images
- Squamous Cell Carcinoma: 322 images
- Healthy Skin (normal + dry + oily): 2,756 images

### Final Dataset
| Split | Images |
|---|---|
| Train | 9,890 |
| Validation | 1,979 |
| Test | 1,346 |
| **Total** | **13,215** |

---

## 10 Classes and Cancer Risk Weights

| Class | Code | Source | Risk Weight |
|---|---|---|---|
| Melanoma | mel | HAM10000 | 1.00 |
| Squamous Cell Carcinoma | scc | Kaggle | 0.90 |
| Basal Cell Carcinoma | bcc | HAM10000 | 0.80 |
| Actinic Keratosis | akiec | HAM10000 | 0.65 |
| Benign Keratosis | bkl | HAM10000 | 0.10 |
| Melanocytic Nevi | nv | HAM10000 | 0.10 |
| Dermatofibroma | df | HAM10000 | 0.05 |
| Vascular Lesion | vasc | HAM10000 | 0.05 |
| Tinea/Ringworm | tinea | Kaggle | 0.00 |
| Healthy Skin | normal | Kaggle | 0.00 |

**Cancer Risk Score** = Σ(P(class_i) × weight_i) × 100

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
  - Freeze ViT encoder
  - Train classification head only
  - Learning rate: 1e-3

Phase 2 (25 epochs):
  - Unfreeze all layers
  - Full fine-tuning
  - Learning rate: 5e-5
  - CosineAnnealingLR scheduler
  - Label smoothing: 0.1
```

---

## Repository Structure

```
Deep-Learning-Final-Project-Group-4/
├── Group-Proposal/
│   └── Group-Proposal.pdf
├── Final-Group-Project-Report/
│   └── (to be added)
├── Final-Group-Presentation/
│   └── (to be added)
└── Code/
    ├── preprocessing/
    │   ├── dataset.py          # Dataset class, transforms, data loading
    │   └── transforms.py       # Augmentation pipeline
    ├── training/
    │   ├── train_vit.py        # ViT fine-tuning (Phase 1 + Phase 2)
    │   └── train_efficientnet.py # EfficientNet-B3 baseline
    ├── models/
    │   └── saved/              # Saved model weights (not on GitHub)
    ├── evaluation/
    │   └── evaluate.py         # Metrics, confusion matrix, attention maps
    ├── app/
    │   └── app.py              # Streamlit web application
    └── notebooks/
        └── eda.ipynb           # Exploratory data analysis
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
```

### 3. Install PyTorch (CUDA 11.8)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Download Data
```bash
# Requires Kaggle API credentials
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p Code/data/raw/ham10000 --unzip
kaggle datasets download -d haroonalam16/20-skin-diseases-dataset -p Code/data/raw/kaggle_diseases --unzip
kaggle datasets download -d shakyadissanayake/oily-dry-and-normal-skin-types-dataset -p Code/data/raw/kaggle_diseases --unzip
```

### 5. Train Models
```bash
cd Code

# Train EfficientNet-B3 baseline
python3 training/train_efficientnet.py

# Train ViT (main model)
python3 training/train_vit.py
```

### 6. Run Streamlit App
```bash
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

---

## Preprocessing Pipeline

1. **Center crop** — Square crop to remove borders
2. **CLAHE** — Contrast Limited Adaptive Histogram Equalization
3. **Resize** — 224×224 for ViT input
4. **Augmentation** (training only):
   - Random rotate, flip, shift/scale
   - Elastic transform, grid distortion
   - Color jitter, Gaussian noise/blur
   - Coarse dropout
5. **Normalization** — ImageNet mean/std

---

## Streamlit App

The web application provides:
- **Live camera capture** via `st.camera_input()`
- **Image upload** for saved images
- **Cancer Risk Score** (0–100 gauge)
- **Class probability bars** for all 10 classes
- **Attention map overlay** showing model focus regions
- **Risk level** (Low / Moderate / Elevated / High)

### Risk Level Thresholds
| Score | Level | Color |
|---|---|---|
| 0–20 | Low Risk | Green |
| 20–50 | Moderate Risk | Amber |
| 50–75 | Elevated Risk | Orange |
| 75–100 | High Risk | Red |

---

## AWS Training Environment

```
Instance:  g5.xlarge
GPU:       NVIDIA A10G (23.6 GB VRAM)
CUDA:      11.8
PyTorch:   2.7.1+cu118
Python:    3.12.3
OS:        Ubuntu 24.04
```

---

## Dependencies

```
torch>=2.7.0
torchvision
transformers>=4.30.0
albumentations
opencv-python
Pillow
scikit-learn
pandas
numpy
matplotlib
seaborn
streamlit
kagglehub
tqdm
wandb
timm
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
  year={2018},
  publisher={Nature Publishing Group}
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
| Member 1 | Dataset pipeline, ViT fine-tuning, EfficientNet baseline, cancer risk score, attention maps, model evaluation |
| Member 2 | EDA, data augmentation, Streamlit app, demo preparation, report sections 1–2 |

---

## License

This project is for educational purposes as part of GWU DATS 6303 Deep Learning course.  
HAM10000 dataset is licensed under CC-BY-NC-SA-4.0.

---

*GWU DATS 6303 Deep Learning | Spring 2026*# Deep-Learning-Final-Project-Group-4
