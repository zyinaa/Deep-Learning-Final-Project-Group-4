# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
import torch
import torchvision.transforms as T
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Class Configuration ───────────────────────────────────
CLASS_INFO = {
    "nv":     {"name": "Melanocytic Nevi",        "weight": 0.10, "idx": 0},
    "mel":    {"name": "Melanoma",                "weight": 1.00, "idx": 1},
    "bkl":    {"name": "Benign Keratosis",        "weight": 0.10, "idx": 2},
    "bcc":    {"name": "Basal Cell Carcinoma",    "weight": 0.80, "idx": 3},
    "akiec":  {"name": "Actinic Keratosis",       "weight": 0.65, "idx": 4},
    "vasc":   {"name": "Vascular Lesion",         "weight": 0.05, "idx": 5},
    "df":     {"name": "Dermatofibroma",          "weight": 0.05, "idx": 6},
    "tinea":  {"name": "Tinea/Ringworm",          "weight": 0.00, "idx": 7},
    "scc":    {"name": "Squamous Cell Carcinoma", "weight": 0.90, "idx": 8},
    "normal": {"name": "Healthy Skin",            "weight": 0.00, "idx": 9},
}

LABEL_TO_IDX = {k: v["idx"] for k, v in CLASS_INFO.items()}
IDX_TO_LABEL = {v["idx"]: k for k, v in CLASS_INFO.items()}
NUM_CLASSES   = len(CLASS_INFO)

# Kaggle dataset folder mapping
KAGGLE_CLASS_MAP = {
    "tinea":  "Tinea Ringworm Candidiasis and other Fungal Infections",
    "scc":    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
}

# Normal skin folders (dry + oily + normal combined)
NORMAL_SKIN_FOLDERS = ["normal", "dry", "oily"]


def compute_cancer_risk(probs: dict) -> float:
    raw = sum(
        probs[cls] * CLASS_INFO[cls]["weight"]
        for cls in probs if cls in CLASS_INFO
    )
    return round(raw * 100, 1)


def build_image_lookup(img_dirs: list) -> dict:
    lookup = {}
    for d in img_dirs:
        if not os.path.exists(d):
            continue
        for fname in os.listdir(d):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_id = os.path.splitext(fname)[0]
                lookup[img_id] = os.path.join(d, fname)
    return lookup


def apply_clahe(img_arr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def preprocess_image(img_path: str, size: int = 224) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img_arr = np.array(img)
    h, w = img_arr.shape[:2]
    size_crop = min(h, w)
    top  = (h - size_crop) // 2
    left = (w - size_crop) // 2
    img_arr = img_arr[top:top+size_crop, left:left+size_crop]
    img_arr = apply_clahe(img_arr)
    img_arr = cv2.resize(img_arr, (size, size),
                         interpolation=cv2.INTER_LANCZOS4)
    return img_arr


def get_transforms(is_train: bool = True):
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if is_train:
        return A.Compose([
            A.Resize(224, 224),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(p=0.3),
                A.GridDistortion(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.3),
                A.GaussianBlur(p=0.3),
            ], p=0.3),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                p=0.3
            ),
            normalize,
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            normalize,
            ToTensorV2(),
        ])


def load_kaggle_data(kaggle_base: str) -> pd.DataFrame:
    """
    Load supplementary Kaggle skin disease images.
    Returns dataframe with image_path and dx columns.
    """
    rows = []

    # Load tinea and scc from Dataset folder
    dataset_train = os.path.join(kaggle_base, "Dataset", "train")
    for cls_key, folder_name in KAGGLE_CLASS_MAP.items():
        folder_path = os.path.join(dataset_train, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found")
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({
                    "image_path": os.path.join(folder_path, fname),
                    "dx": cls_key,
                    "source": "kaggle"
                })
        print(f"Loaded {cls_key}: {len([r for r in rows if r['dx']==cls_key])} images")

    # Load normal skin (dry + oily + normal combined)
    skin_types_train = os.path.join(
        kaggle_base, "Oily-Dry-Skin-Types", "train"
    )
    normal_count = 0
    for folder in NORMAL_SKIN_FOLDERS:
        folder_path = os.path.join(skin_types_train, folder)
        if not os.path.exists(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({
                    "image_path": os.path.join(folder_path, fname),
                    "dx": "normal",
                    "source": "kaggle"
                })
                normal_count += 1
    print(f"Loaded normal: {normal_count} images")

    return pd.DataFrame(rows)


class SkinDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        img_dirs: list,
        kaggle_base: str,
        transform=None,
        split: str = "train",
        val_size: float = 0.15,
        test_size: float = 0.10,
        random_state: int = 42
    ):
        self.transform = transform
        self.split = split

        # ── Load HAM10000 ─────────────────────────────────
        ham_df = pd.read_csv(metadata_path)
        self.img_lookup = build_image_lookup(img_dirs)
        ham_df = ham_df[
            ham_df["image_id"].isin(self.img_lookup.keys())
        ].copy()
        ham_df = ham_df[
            ham_df["dx"].isin(LABEL_TO_IDX.keys())
        ].copy()
        ham_df = ham_df.reset_index(drop=True)

        # Add image_path column for unified access
        ham_df["image_path"] = ham_df["image_id"].map(self.img_lookup)
        ham_df["source"] = "ham10000"

        # Patient-level split for HAM10000
        group_col = "lesion_id" if "lesion_id" in ham_df.columns else "image_id"

        splitter1 = GroupShuffleSplit(
            n_splits=1, test_size=test_size,
            random_state=random_state
        )
        train_val_idx, test_idx = next(
            splitter1.split(ham_df, groups=ham_df[group_col])
        )
        ham_trainval = ham_df.iloc[train_val_idx].reset_index(drop=True)
        ham_test     = ham_df.iloc[test_idx].reset_index(drop=True)

        val_ratio = val_size / (1 - test_size)
        splitter2 = GroupShuffleSplit(
            n_splits=1, test_size=val_ratio,
            random_state=random_state
        )
        train_idx, val_idx = next(
            splitter2.split(ham_trainval,
                           groups=ham_trainval[group_col])
        )
        ham_train = ham_trainval.iloc[train_idx].reset_index(drop=True)
        ham_val   = ham_trainval.iloc[val_idx].reset_index(drop=True)

        # ── Load Kaggle data ──────────────────────────────
        kaggle_df = load_kaggle_data(kaggle_base)

        # Split Kaggle data 75/15/10
        from sklearn.model_selection import train_test_split
        kaggle_trainval, kaggle_test = train_test_split(
            kaggle_df, test_size=test_size,
            random_state=random_state,
            stratify=kaggle_df["dx"]
        )
        kaggle_train, kaggle_val = train_test_split(
            kaggle_trainval,
            test_size=val_ratio,
            random_state=random_state,
            stratify=kaggle_trainval["dx"]
        )

        # ── Combine HAM10000 + Kaggle ─────────────────────
        split_map = {
            "train": pd.concat([
                ham_train[["image_path", "dx", "source"]],
                kaggle_train[["image_path", "dx", "source"]]
            ], ignore_index=True),
            "val": pd.concat([
                ham_val[["image_path", "dx", "source"]],
                kaggle_val[["image_path", "dx", "source"]]
            ], ignore_index=True),
            "test": pd.concat([
                ham_test[["image_path", "dx", "source"]],
                kaggle_test[["image_path", "dx", "source"]]
            ], ignore_index=True),
        }

        self.df = split_map[split].reset_index(drop=True)

        print(f"\n{split} set: {len(self.df)} images")
        print("Class distribution:")
        for cls, count in self.df["dx"].value_counts().items():
            bar = chr(9608) * (count // 100)
            print(f"  {cls:10} {count:5,}  {bar}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        label    = LABEL_TO_IDX[row["dx"]]
        img_path = row["image_path"]

        img_arr = preprocess_image(img_path)

        if self.transform:
            augmented  = self.transform(image=img_arr)
            img_tensor = augmented["image"]
        else:
            img_tensor = T.ToTensor()(Image.fromarray(img_arr))

        return img_tensor, label

    def get_class_weights(self) -> torch.Tensor:
        counts  = self.df["dx"].value_counts()
        weights = []
        for _, row in self.df.iterrows():
            weights.append(1.0 / counts[row["dx"]])
        return torch.tensor(weights, dtype=torch.float)


def get_dataloaders(
    metadata_path: str,
    img_dirs: list,
    kaggle_base: str,
    batch_size: int = 32,
    num_workers: int = 4
):
    train_dataset = SkinDataset(
        metadata_path=metadata_path,
        img_dirs=img_dirs,
        kaggle_base=kaggle_base,
        transform=get_transforms(is_train=True),
        split="train"
    )
    val_dataset = SkinDataset(
        metadata_path=metadata_path,
        img_dirs=img_dirs,
        kaggle_base=kaggle_base,
        transform=get_transforms(is_train=False),
        split="val"
    )
    test_dataset = SkinDataset(
        metadata_path=metadata_path,
        img_dirs=img_dirs,
        kaggle_base=kaggle_base,
        transform=get_transforms(is_train=False),
        split="test"
    )

    class_weights = train_dataset.get_class_weights()
    sampler = WeightedRandomSampler(
        weights=class_weights,
        num_samples=len(class_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    METADATA = os.path.join(
        BASE, "data/raw/ham10000/HAM10000_metadata.csv"
    )
    IMG_DIRS = [
        os.path.join(BASE, "data/raw/ham10000/HAM10000_images_part_1"),
        os.path.join(BASE, "data/raw/ham10000/HAM10000_images_part_2"),
    ]
    KAGGLE_BASE = os.path.join(BASE, "data/raw/kaggle_diseases")

    print("=" * 50)
    print("Testing SkinDataset with all sources")
    print("=" * 50)

    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_path=METADATA,
        img_dirs=IMG_DIRS,
        kaggle_base=KAGGLE_BASE,
        batch_size=32,
        num_workers=2
    )

    imgs, labels = next(iter(train_loader))
    print(f"\nBatch shape: {imgs.shape}")
    print(f"Labels: {[IDX_TO_LABEL[l.item()] for l in labels[:8]]}")
    print("\nDataset test passed!")
