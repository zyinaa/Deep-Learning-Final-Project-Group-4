# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.dataset import SkinDataset, CLASS_INFO, IDX_TO_LABEL, NUM_CLASSES
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models/saved/vit_best.pth")
OUTPUT_DIR  = os.path.join(BASE_DIR, "evaluation/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METADATA_PATH = os.path.join(BASE_DIR, "data/raw/ham10000/HAM10000_metadata.csv")
IMG_DIRS = [
    os.path.join(BASE_DIR, "data/raw/ham10000/HAM10000_images_part_1"),
    os.path.join(BASE_DIR, "data/raw/ham10000/HAM10000_images_part_2"),
]
KAGGLE_BASE = os.path.join(BASE_DIR, "data/raw/kaggle_diseases")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load model
print("Loading model...")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Load test set
print("Loading test set...")
test_dataset = SkinDataset(
    metadata_path=METADATA_PATH,
    img_dirs=IMG_DIRS,
    kaggle_base=KAGGLE_BASE,
    split="test"
)
test_loader = DataLoader(test_dataset, batch_size=32,
                        shuffle=False, num_workers=4)

# Get probabilities
all_probs  = []
all_labels = []

print("Running inference...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

# Find optimal threshold for each high-risk class
HIGH_RISK_CLASSES = {
    "mel":   1,
    "bcc":   3,
    "scc":   8,
    "akiec": 4,
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("#0A1628")
fig.suptitle("ROC Curves — High Risk Classes", 
             color="white", fontsize=14, fontweight="bold")

results = {}

for ax, (cls_name, cls_idx) in zip(axes.flatten(), HIGH_RISK_CLASSES.items()):
    # Binary labels for this class
    binary_labels = (all_labels == cls_idx).astype(int)
    class_probs   = all_probs[:, cls_idx]

    # ROC curve
    fpr, tpr, thresholds = roc_curve(binary_labels, class_probs)
    roc_auc = auc(fpr, tpr)

    # Youden's J = TPR - FPR (maximize this)
    j_scores  = tpr - fpr
    best_idx  = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    best_tpr    = tpr[best_idx]
    best_fpr    = fpr[best_idx]

    results[cls_name] = {
        "best_threshold": float(best_thresh),
        "tpr_at_threshold": float(best_tpr),
        "fpr_at_threshold": float(best_fpr),
        "auc": float(roc_auc)
    }

    # Plot
    ax.set_facecolor("#0D1F3C")
    ax.plot(fpr, tpr, color="#00B4D8", linewidth=2,
            label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#4A5568",
            linestyle="--", linewidth=1)
    ax.scatter([best_fpr], [best_tpr], color="#FF2D2D",
              s=100, zorder=5,
              label=f"Best threshold = {best_thresh:.3f}")
    ax.set_title(CLASS_INFO[cls_name]["name"],
                color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate", color="white", fontsize=10)
    ax.set_ylabel("True Positive Rate", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#0D1F3C", labelcolor="white", fontsize=9)
    ax.spines["bottom"].set_color("#1E3A5F")
    ax.spines["left"].set_color("#1E3A5F")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0A1628")
plt.close()
print(f"Saved: {save_path}")

# Print results
print("\n" + "="*50)
print("OPTIMAL THRESHOLDS (Youden's J)")
print("="*50)
for cls_name, res in results.items():
    print(f"\n{CLASS_INFO[cls_name]['name']}:")
    print(f"  Best threshold: {res['best_threshold']:.4f}")
    print(f"  TPR (Recall):   {res['tpr_at_threshold']:.4f}")
    print(f"  FPR:            {res['fpr_at_threshold']:.4f}")
    print(f"  AUC:            {res['auc']:.4f}")

print("\nDone!")
