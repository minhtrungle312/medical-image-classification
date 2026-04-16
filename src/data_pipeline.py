"""
Data Pipeline for ISIC Skin Cancer Dataset.

Handles dataset loading, preprocessing, augmentation, and
class imbalance management for dermoscopic image classification.

Dataset source: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ISIC 9-class mapping (sorted alphabetically to match folder order)
CLASS_NAMES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion",
]

# Malignant / high-risk classes for clinical flagging
MALIGNANT_CLASSES = {"melanoma", "basal cell carcinoma", "squamous cell carcinoma"}

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224


class ISICSkinDataset(Dataset):
    """ISIC Skin Cancer dataset for dermoscopic image classification."""

    def __init__(
        self,
        image_paths: list,
        labels: list,
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(split: str = "train") -> transforms.Compose:
    """Get data transforms for a given split with augmentation for training."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def download_dataset(data_dir: str = "data") -> str:
    """
    Download the ISIC Skin Cancer dataset from Kaggle using kagglehub.

    Returns:
        Path to the downloaded dataset root directory.
    """
    import kagglehub

    logger.info("Downloading ISIC Skin Cancer dataset from Kaggle...")
    path = kagglehub.dataset_download("nodoubttome/skin-cancer9-classesisic")
    logger.info(f"Dataset downloaded to: {path}")
    return path


def load_isic_dataset(data_dir: str) -> Tuple[list, list]:
    """
    Load ISIC dataset from a folder-based directory structure.

    Expected structure (Kaggle: nodoubttome/skin-cancer9-classesisic):
        data_dir/
            Skin cancer ISIC The International Skin Imaging Collaboration/
                Train/
                    actinic keratosis/
                    basal cell carcinoma/
                    ...
                Test/
                    actinic keratosis/
                    ...

    Falls back to flat Train/Test structure if the nested folder exists directly.

    Returns:
        image_paths: list of image file paths
        labels: list of integer labels
    """
    data_path = Path(data_dir)

    # Auto-detect nested Kaggle structure
    nested = data_path / "Skin cancer ISIC The International Skin Imaging Collaboration"
    if nested.exists():
        data_path = nested

    # Locate Train and/or Test directories (case-insensitive search)
    train_dir = None
    test_dir = None
    for child in data_path.iterdir():
        if child.is_dir() and child.name.lower() == "train":
            train_dir = child
        elif child.is_dir() and child.name.lower() == "test":
            test_dir = child

    if train_dir is None and test_dir is None:
        raise FileNotFoundError(
            f"No Train/ or Test/ directory found in {data_path}. "
            f"Contents: {[c.name for c in data_path.iterdir()]}"
        )

    image_paths = []
    labels = []

    # Build class-to-index mapping from CLASS_NAMES
    class_to_idx = {name.lower(): idx for idx, name in enumerate(CLASS_NAMES)}

    for split_dir in [train_dir, test_dir]:
        if split_dir is None:
            continue
        for class_folder in sorted(split_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            class_name = class_folder.name.lower().strip()
            if class_name not in class_to_idx:
                logger.warning(f"Unknown class folder '{class_folder.name}', skipping")
                continue
            label = class_to_idx[class_name]
            for img_file in class_folder.iterdir():
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    image_paths.append(str(img_file))
                    labels.append(label)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {data_path}")

    logger.info(f"Loaded {len(image_paths)} images with {len(set(labels))} classes")
    return image_paths, labels


def split_dataset(
    image_paths: list,
    labels: list,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Tuple[list, list]]:
    """Split dataset into train/validation/test sets with stratification."""
    # First split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_fraction,
        stratify=train_val_labels,
        random_state=random_state,
    )

    logger.info(
        f"Split sizes - Train: {len(train_paths)}, "
        f"Val: {len(val_paths)}, Test: {len(test_paths)}"
    )

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def compute_class_weights(labels: list) -> torch.Tensor:
    """Compute inverse frequency class weights for handling imbalance."""
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = len(labels)
    weights = total / (NUM_CLASSES * class_counts.astype(float) + 1e-6)
    weights = torch.FloatTensor(weights)
    logger.info(f"Class weights: {weights.tolist()}")
    return weights


def get_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """Create weighted random sampler for handling class imbalance."""
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts.astype(float) + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
    return sampler


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create train/val/test data loaders with preprocessing and augmentation.

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    image_paths, labels = load_isic_dataset(data_dir)
    splits = split_dataset(image_paths, labels)

    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    # Compute class weights from training set
    class_weights = compute_class_weights(train_labels)

    # Create datasets with appropriate transforms
    train_dataset = ISICSkinDataset(train_paths, train_labels, get_transforms("train"))
    val_dataset = ISICSkinDataset(val_paths, val_labels, get_transforms("val"))
    test_dataset = ISICSkinDataset(test_paths, test_labels, get_transforms("test"))

    # Weighted sampling for training
    train_sampler = get_weighted_sampler(train_labels) if use_weighted_sampling else None

    # Only use pin_memory on CUDA (not supported on MPS/CPU)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    logger.info(
        f"DataLoaders created - "
        f"Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}, "
        f"Test batches: {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader, class_weights


def validate_data_quality(data_dir: str) -> Dict[str, any]:
    """Run data quality checks on the dataset."""
    results = {"passed": True, "checks": {}}

    try:
        image_paths, labels = load_isic_dataset(data_dir)
    except (FileNotFoundError, ValueError) as e:
        results["passed"] = False
        results["checks"]["load"] = {"passed": False, "error": str(e)}
        return results

    # Check: minimum dataset size
    min_size = 100
    size_ok = len(image_paths) >= min_size
    results["checks"]["min_size"] = {
        "passed": size_ok,
        "actual": len(image_paths),
        "expected": f">= {min_size}",
    }
    if not size_ok:
        results["passed"] = False

    # Check: all classes represented
    unique_labels = set(labels)
    all_classes = len(unique_labels) == NUM_CLASSES
    results["checks"]["all_classes"] = {
        "passed": all_classes,
        "actual": len(unique_labels),
        "expected": NUM_CLASSES,
    }
    if not all_classes:
        results["passed"] = False

    # Check: no corrupt images (sample check)
    sample_size = min(50, len(image_paths))
    sample_indices = np.random.choice(len(image_paths), sample_size, replace=False)
    corrupt_count = 0
    for idx in sample_indices:
        try:
            img = Image.open(image_paths[idx])
            img.verify()
        except Exception:
            corrupt_count += 1

    results["checks"]["image_integrity"] = {
        "passed": corrupt_count == 0,
        "corrupt_in_sample": corrupt_count,
        "sample_size": sample_size,
    }
    if corrupt_count > 0:
        results["passed"] = False

    # Check: class distribution (no class < 1% of total)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    min_ratio = class_counts.min() / len(labels)
    results["checks"]["class_balance"] = {
        "passed": True,  # Informational, imbalance is expected
        "distribution": {CLASS_NAMES[i]: int(c) for i, c in enumerate(class_counts)},
        "min_class_ratio": float(min_ratio),
        "note": "Class imbalance handled via weighted sampling/loss",
    }

    return results
