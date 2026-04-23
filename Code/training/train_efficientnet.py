# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.dataset import (
    get_dataloaders, IDX_TO_LABEL, NUM_CLASSES
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "metadata": os.path.join(BASE, "data/raw/ham10000/HAM10000_metadata.csv"),
    "img_dirs": [
        os.path.join(BASE, "data/raw/ham10000/HAM10000_images_part_1"),
        os.path.join(BASE, "data/raw/ham10000/HAM10000_images_part_2"),
    ],
    "kaggle_base": os.path.join(BASE, "data/raw/kaggle_diseases"),
    "save_dir":    os.path.join(BASE, "models/saved"),
    "batch_size":  32,
    "num_workers": 4,
    "epochs":      20,
    "lr":          1e-4,
    "num_classes": NUM_CLASSES,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)


def build_efficientnet(num_classes):
    model = models.efficientnet_b3(
        weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds  = []
    all_labels = []

    pbar = tqdm(loader, desc="Training")
    for imgs, labels in pbar:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), f1


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating"):
            imgs   = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), f1, all_preds, all_labels


def train():
    print("=" * 50)
    print("EfficientNet-B3 Baseline (10 classes)")
    print("=" * 50)
    print(f"Device:  {CONFIG['device']}")
    print(f"Epochs:  {CONFIG['epochs']}")
    print(f"Classes: {NUM_CLASSES}")

    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_path=CONFIG["metadata"],
        img_dirs=CONFIG["img_dirs"],
        kaggle_base=CONFIG["kaggle_base"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )

    model = build_efficientnet(CONFIG["num_classes"])
    model = model.to(CONFIG["device"])

    class_counts = torch.zeros(NUM_CLASSES)
    for _, labels in train_loader:
        for l in labels:
            class_counts[l.item()] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = class_weights.to(CONFIG["device"])

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"]
    )

    best_val_f1 = 0
    history     = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        print("-" * 40)

        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, CONFIG["device"]
        )
        val_loss, val_f1, _, _ = validate(
            model, val_loader, criterion, CONFIG["device"]
        )
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}  F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f}  F1: {val_f1:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_f1": train_f1,
            "val_loss": val_loss,     "val_f1": val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(),
                os.path.join(CONFIG["save_dir"],
                             "efficientnet_b3_best.pth")
            )
            print(f"Saved! Best Val F1: {best_val_f1:.4f}")

    # Final test evaluation
    print("\n" + "=" * 50)
    print("Final Test Evaluation")
    print("=" * 50)

    model.load_state_dict(torch.load(
        os.path.join(CONFIG["save_dir"], "efficientnet_b3_best.pth")
    ))
    test_loss, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, CONFIG["device"]
    )

    print(f"Test Macro F1: {test_f1:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")

    present_labels = sorted(set(test_labels))
    present_names  = [IDX_TO_LABEL[i] for i in present_labels]
    print(classification_report(
        test_labels, test_preds,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))

    results = {
        "model": "EfficientNet-B3",
        "num_classes": NUM_CLASSES,
        "best_val_f1": best_val_f1,
        "test_f1_macro": test_f1,
        "test_loss": test_loss,
        "history": history,
    }
    with open(
        os.path.join(CONFIG["save_dir"],
                     "efficientnet_results.json"), "w"
    ) as f:
        json.dump(results, f, indent=2)

    print(f"\nBest Val F1: {best_val_f1:.4f}")
    print(f"Test F1:     {test_f1:.4f}")
    print("Training complete!")


if __name__ == "__main__":
    train()
