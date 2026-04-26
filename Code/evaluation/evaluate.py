# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score
)
import seaborn as sns
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.dataset import SkinDataset, CLASS_INFO, IDX_TO_LABEL, NUM_CLASSES
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader

# ── Config ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data/raw")
MODEL_PATH  = os.path.join(BASE_DIR, "models/saved/vit_best.pth")
OUTPUT_DIR  = os.path.join(BASE_DIR, "evaluation/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METADATA_PATH = os.path.join(DATA_DIR, "ham10000/HAM10000_metadata.csv")
IMG_DIRS = [
    os.path.join(DATA_DIR, "ham10000/HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "ham10000/HAM10000_images_part_2"),
]
KAGGLE_BASE = os.path.join(DATA_DIR, "kaggle_diseases")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── Load model ────────────────────────────────────────────
print("Loading model...")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ── Load test set ─────────────────────────────────────────
print("Loading test set...")
test_dataset = SkinDataset(
    metadata_path=METADATA_PATH,
    img_dirs=IMG_DIRS,
    kaggle_base=KAGGLE_BASE,
    split="test"
)
test_loader = DataLoader(test_dataset, batch_size=32,
                        shuffle=False, num_workers=4)
print(f"Test set size: {len(test_dataset)}")

# ── Inference ─────────────────────────────────────────────
all_preds  = []
all_labels = []

print("Running inference...")
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        if i % 5 == 0:
            print(f"  Batch {i}/{len(test_loader)}")

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Metrics ───────────────────────────────────────────────
class_names = [CLASS_INFO[IDX_TO_LABEL[i]]["name"] for i in range(NUM_CLASSES)]

f1_macro    = f1_score(all_labels, all_preds, average="macro")
f1_weighted = f1_score(all_labels, all_preds, average="weighted")
accuracy    = (all_preds == all_labels).mean()

print(f"\nTest Accuracy:      {accuracy*100:.2f}%")
print(f"Test F1 (macro):    {f1_macro:.4f}")
print(f"Test F1 (weighted): {f1_weighted:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ── Confusion Matrix ──────────────────────────────────────
print("Generating confusion matrix...")
cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.patch.set_facecolor("#0A1628")

for ax, data, title, fmt in zip(
    axes,
    [cm, cm_norm],
    ["Confusion Matrix (Counts)", "Confusion Matrix (Normalized)"],
    ["d", ".2f"]
):
    ax.set_facecolor("#0D1F3C")
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor="#0A1628",
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", color="white", fontsize=11)
    ax.set_ylabel("Actual", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0A1628")
plt.close()
print(f"Saved: {save_path}")

# ── Per-class F1 ──────────────────────────────────────────
f1_per_class = f1_score(all_labels, all_preds, average=None)

fig2, ax2 = plt.subplots(figsize=(12, 6))
fig2.patch.set_facecolor("#0A1628")
ax2.set_facecolor("#0D1F3C")

colors = ["#E24B4A" if f < 0.5 else "#EF9F27" if f < 0.7 else "#00CC88"
          for f in f1_per_class]
bars = ax2.barh(class_names, f1_per_class, color=colors, edgecolor="none")

for bar, val in zip(bars, f1_per_class):
    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", color="white", fontsize=10)

ax2.set_xlim(0, 1.15)
ax2.set_xlabel("F1 Score", color="white", fontsize=11)
ax2.set_title("Per-Class F1 Score — ViT-base-patch16-224",
             color="white", fontsize=14, fontweight="bold")
ax2.tick_params(colors="white")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_color("#1E3A5F")
ax2.spines["left"].set_color("#1E3A5F")
ax2.axvline(x=f1_macro, color="#00B4D8", linestyle="--",
           linewidth=1.5, label=f"Macro F1 = {f1_macro:.4f}")
ax2.legend(facecolor="#0D1F3C", labelcolor="white")

plt.tight_layout()
save_path2 = os.path.join(OUTPUT_DIR, "per_class_f1.png")
plt.savefig(save_path2, dpi=150, bbox_inches="tight", facecolor="#0A1628")
plt.close()
print(f"Saved: {save_path2}")
print("\nDone!")
