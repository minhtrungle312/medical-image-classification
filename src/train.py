"""
Training script with MLflow experiment tracking.

Supports training three model architectures:
- Transfer Learning (ResNet50, EfficientNet-B0)
- Vision Transformer (ViT)

Two-stage training:
  Stage 1 — Head-only (backbone frozen) warmup
  Stage 2 — Full fine-tuning with cosine annealing

Two-step workflow:
  Step 1: Train with val split → find best epoch & hyperparams
  Step 2: --full-train on 100% data using the best epoch count
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import numpy as np

# Set MLflow tracking URI (default to local mlruns if not set)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)

from src.data_pipeline import create_data_loaders, NUM_CLASSES
from src.models.transfer_learning import ResNet50Model, EfficientNetModel
from src.models.vit_model import ViTModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance by down-weighting easy examples."""

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction='none',
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_model(model_name: str, num_classes: int = NUM_CLASSES, freeze_backbone: bool = False) -> nn.Module:
    """Factory function to create model by name."""
    models_map = {
        "resnet50": lambda: ResNet50Model(num_classes=num_classes, freeze_backbone=freeze_backbone),
        "efficientnet": lambda: EfficientNetModel(num_classes=num_classes, freeze_backbone=freeze_backbone),
        "vit": lambda: ViTModel(num_classes=num_classes, freeze_backbone=freeze_backbone),
    }

    if model_name not in models_map:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(models_map.keys())}"
        )

    return models_map[model_name]()


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return {"loss": epoch_loss, "accuracy": epoch_acc}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)
    metrics = {
        "loss": epoch_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "f2": fbeta_score(
            all_labels, all_preds, beta=2, average="macro", zero_division=0
        ),
    }

    return metrics


def train_model(
    model_name: str,
    data_dir: str,
    output_dir: str = "models",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    num_workers: int = 4,
    experiment_name: str = "skin-cancer-classification",
    full_train: bool = False,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Full training pipeline with MLflow tracking and two-stage training.

    Args:
        model_name: One of 'resnet50', 'efficientnet', 'vit'
        data_dir: Path to ISIC dataset
        output_dir: Directory to save trained models
        epochs: Number of training epochs (fine-tuning stage)
        batch_size: Batch size for data loaders
        learning_rate: Initial learning rate for fine-tuning stage
        weight_decay: L2 regularization weight
        focal_gamma: Focal Loss gamma parameter
        label_smoothing: Label smoothing factor
        num_workers: Number of data loader workers
        experiment_name: MLflow experiment name
        full_train: If True, train on 100% of training data (no val split).
                     Model is saved at the last epoch.

    Returns:
        Trained model and final/best metrics
    """
    # Use MPS (Apple Silicon) if available, then CUDA, then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Training {model_name} on {device} with lr={learning_rate}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Data loaders
    train_loader, val_loader, _, class_weights = create_data_loaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        full_train=full_train,
    )

    if full_train:
        logger.info("Full train mode: using 100% of training data, no validation split")

    # Loss with class weights — Focal Loss to focus on hard/minority examples
    class_weights = class_weights.to(device)
    criterion = FocalLoss(weight=class_weights, gamma=focal_gamma, label_smoothing=label_smoothing)

    # ── Stage 1: Train only the classification head (backbone frozen) ──
    warmup_epochs = min(5, epochs // 3)
    model = get_model(model_name, freeze_backbone=True)
    model = model.to(device)

    optimizer_head = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate * 10,
        weight_decay=weight_decay,
    )

    logger.info(f"Stage 1: Training head only for {warmup_epochs} epochs (lr={learning_rate * 10})")
    for epoch in range(1, warmup_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer_head, device)
        if val_loader is not None:
            val_metrics = validate(model, val_loader, criterion, device)
            logger.info(
                f"[Head] Epoch {epoch}/{warmup_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
        else:
            logger.info(
                f"[Head] Epoch {epoch}/{warmup_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}"
            )

    # ── Stage 2: Unfreeze all layers, fine-tune end-to-end ──
    logger.info("Stage 2: Unfreezing backbone for full fine-tuning")
    model.unfreeze_all()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # MLflow tracking
    mlflow.set_experiment(experiment_name)

    best_val_metrics = None
    best_val_f1 = 0.0

    with mlflow.start_run(run_name=model_name):
        # Log hyperparameters
        mlflow.log_params(
            {
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "focal_gamma": focal_gamma,
                "label_smoothing": label_smoothing,
                "full_train": full_train,
                "warmup_epochs": warmup_epochs,
                "optimizer": "AdamW",
                "loss": f"FocalLoss (weighted, gamma={focal_gamma})",
                "training_strategy": "two-stage (frozen head -> full fine-tune)",
                "device": str(device),
                "num_params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }
        )

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validate (skip if full_train and no val set)
            val_metrics = None
            if val_loader is not None:
                val_metrics = validate(model, val_loader, criterion, device)

            # Step scheduler
            scheduler.step(epoch)

            epoch_time = time.time() - start_time

            # Log metrics
            log_dict = {
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "epoch_time": epoch_time,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if val_metrics:
                log_dict.update({
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "val_f2": val_metrics["f2"],
                })
            mlflow.log_metrics(log_dict, step=epoch)

            if val_metrics:
                logger.info(
                    f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s) - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Val Recall: {val_metrics['recall']:.4f}, "
                    f"Val F1: {val_metrics['f1']:.4f}, "
                    f"Val F2: {val_metrics['f2']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s) - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}"
                )

            if full_train:
                # In full_train mode: save model at every epoch (overwrite)
                best_val_metrics = train_metrics
                model_path = os.path.join(output_dir, f"{model_name}_best.pth")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": model_name,
                        "epoch": epoch,
                        "train_metrics": train_metrics,
                        "num_classes": NUM_CLASSES,
                    },
                    model_path,
                )
            elif val_metrics and val_metrics["f1"] > best_val_f1:
                # Save best model by val_f1
                best_val_f1 = val_metrics["f1"]
                best_val_metrics = val_metrics
                model_path = os.path.join(output_dir, f"{model_name}_best.pth")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": model_name,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "num_classes": NUM_CLASSES,
                    },
                    model_path,
                )
                logger.info(f"Saved best model to {model_path} (val_f1={best_val_f1:.4f})")

        # Log best/final metrics
        if best_val_metrics:
            mlflow.log_metrics({f"best_{k}": v for k, v in best_val_metrics.items()})

        # Log model artifact
        model_path = os.path.join(output_dir, f"{model_name}_best.pth")
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path)

        # Log model with MLflow
        mlflow.pytorch.log_model(model, f"{model_name}_model")

    logger.info(f"Training complete. Best metrics: {best_val_metrics}")
    return model, best_val_metrics


def train_all_models(
    data_dir: str,
    output_dir: str = "models",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = None,
    full_train: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Train all models and return their metrics for comparison."""
    model_names = ["resnet50", "efficientnet", "vit"]
    # Per-model learning rates for full fine-tuning
    lr_map = {"resnet50": 1e-4, "efficientnet": 1e-4, "vit": 5e-5}
    results = {}

    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model: {model_name}")
        logger.info(f"{'='*60}\n")

        lr = (
            learning_rate if learning_rate is not None else lr_map.get(model_name, 1e-4)
        )

        _, metrics = train_model(
            model_name=model_name,
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            full_train=full_train,
        )
        results[model_name] = metrics

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train skin cancer classification models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["resnet50", "efficientnet", "vit", "all"],
        help="Model to train",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Path to ISIC dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Train on 100%% of training data (no val split). Use after finding best epoch.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Data loader workers"
    )

    args = parser.parse_args()

    if args.model == "all":
        results = train_all_models(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            full_train=args.full_train,
        )
        print("\n=== Training Results ===")
        for name, metrics in results.items():
            print(f"\n{name}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
    else:
        train_model(
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_workers=args.num_workers,
            full_train=args.full_train,
        )
