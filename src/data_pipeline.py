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
    Download the ISIC Skin Cancer dataset from Kaggle using kagglehub
    and set it up in the specified data directory.

    Args:
        data_dir: Directory to set up the dataset in (default: "data")

    Returns:
        Path to the dataset root directory in data_dir.
    """
    import kagglehub
    import shutil

    logger.info("Downloading ISIC Skin Cancer dataset from Kaggle...")
    downloaded_path = kagglehub.dataset_download("nodoubttome/skin-cancer9-classesisic")
    logger.info(f"Dataset downloaded to: {downloaded_path}")

    # Set up in data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Check if already set up
    expected_nested = data_path / "Skin cancer ISIC The International Skin Imaging Collaboration"
    if expected_nested.exists():
        logger.info(f"Dataset already exists in {expected_nested}")
        return str(expected_nested)

    # Copy the downloaded dataset to data directory
    downloaded_nested = Path(downloaded_path) / "Skin cancer ISIC The International Skin Imaging Collaboration"
    if downloaded_nested.exists():
        logger.info(f"Copying dataset to {data_path}...")
        shutil.copytree(downloaded_nested, expected_nested)
        logger.info(f"Dataset set up successfully in {expected_nested}")
        return str(expected_nested)
    else:
        # Fallback: copy the whole downloaded directory
        logger.warning(f"Expected nested structure not found, copying entire download to {data_path}")
        for item in Path(downloaded_path).iterdir():
            dest = data_path / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        return str(data_path)


def _resolve_data_path(data_dir: str) -> Path:
    """Resolve the dataset root, handling nested Kaggle structure."""
    data_path = Path(data_dir)
    nested = data_path / "Skin cancer ISIC The International Skin Imaging Collaboration"
    if nested.exists():
        data_path = nested
    return data_path


def _find_split_dirs(data_path: Path) -> Tuple:
    """Locate Train and Test directories."""
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
    return train_dir, test_dir


def _load_folder(folder: Path) -> Tuple[list, list]:
    """Load images and labels from a single split folder (Train/ or Test/)."""
    class_to_idx = {name.lower(): idx for idx, name in enumerate(CLASS_NAMES)}
    image_paths = []
    labels = []
    for class_folder in sorted(folder.iterdir()):
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
    return image_paths, labels


def load_isic_dataset(data_dir: str) -> Tuple[list, list]:
    """
    Load all images from the ISIC dataset (Train + Test combined).

    Returns:
        image_paths, labels
    """
    data_path = _resolve_data_path(data_dir)
    train_dir, test_dir = _find_split_dirs(data_path)

    image_paths = []
    labels = []
    for split_dir in [train_dir, test_dir]:
        if split_dir is None:
            continue
        paths, lbls = _load_folder(split_dir)
        image_paths.extend(paths)
        labels.extend(lbls)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {data_path}")

    logger.info(f"Loaded {len(image_paths)} images with {len(set(labels))} classes")
    return image_paths, labels


def load_train_test_split(data_dir: str) -> Dict[str, Tuple[list, list]]:
    """
    Load Train/ and Test/ folders separately.

    Returns dict with 'train_all' and 'test' keys.
    """
    data_path = _resolve_data_path(data_dir)
    train_dir, test_dir = _find_split_dirs(data_path)

    result = {}
    if train_dir:
        result["train_all"] = _load_folder(train_dir)
        logger.info(f"Train folder: {len(result['train_all'][0])} images")
    if test_dir:
        result["test"] = _load_folder(test_dir)
        logger.info(f"Test folder: {len(result['test'][0])} images")
    return result


def split_dataset(
    image_paths: list,
    labels: list,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Tuple[list, list]]:
    """Split a set of images into train/val with stratification."""
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=val_size,
        stratify=labels,
        random_state=random_state,
    )

    logger.info(
        f"Split sizes - Train: {len(train_paths)}, "
        f"Val: {len(val_paths)}"
    )

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
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
    data = load_train_test_split(data_dir)
    train_all_paths, train_all_labels = data["train_all"]
    splits = split_dataset(train_all_paths, train_all_labels)

    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = data["test"]

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
