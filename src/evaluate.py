"""
Model evaluation and comparison module.

Evaluates all trained models on the test set, generates metrics,
confusion matrices, ROC curves, and a comparison table.
Focuses on recall as the priority metric for medical diagnosis.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import mlflow

from src.data_pipeline import (
    create_data_loaders,
    CLASS_NAMES,
    NUM_CLASSES,
)
from src.train import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on test set.

    Returns:
        all_labels: Ground truth labels
        all_preds: Predicted labels
        all_probs: Predicted probabilities (for ROC-AUC)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(
    labels: np.ndarray, preds: np.ndarray, probs: np.ndarray
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    # Binarize labels for ROC-AUC
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f2_macro": fbeta_score(labels, preds, beta=2, average="macro", zero_division=0),
        "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "f2_weighted": fbeta_score(labels, preds, beta=2, average="weighted", zero_division=0),
    }

    # ROC-AUC (One-vs-Rest)
    try:
        metrics["roc_auc_macro"] = roc_auc_score(
            labels_bin, probs, average="macro", multi_class="ovr"
        )
        metrics["roc_auc_weighted"] = roc_auc_score(
            labels_bin, probs, average="weighted", multi_class="ovr"
        )
    except ValueError:
        metrics["roc_auc_macro"] = 0.0
        metrics["roc_auc_weighted"] = 0.0

    # Per-class recall (important for medical diagnosis)
    per_class_recall = recall_score(labels, preds, average=None, zero_division=0)
    for i, cls_name in enumerate(CLASS_NAMES):
        if i < len(per_class_recall):
            metrics[f"recall_{cls_name}"] = per_class_recall[i]

    return metrics


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    model_name: str,
    output_dir: str,
) -> str:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[0],
    )
    axes[0].set_title(f"{model_name} - Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[1],
    )
    axes[1].set_title(f"{model_name} - Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix to {path}")
    return path


def plot_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    output_dir: str,
) -> str:
    """Plot and save ROC curves for each class."""
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, cls_name in enumerate(CLASS_NAMES):
        if i < labels_bin.shape[1] and i < probs.shape[1]:
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{cls_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} - ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"roc_curves_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curves to {path}")
    return path


def create_comparison_table(
    all_results: Dict[str, Dict[str, float]],
    output_dir: str,
) -> pd.DataFrame:
    """Create and save model comparison table."""
    # Select key metrics for comparison
    key_metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_macro",
        "recall_melanoma",  # Melanoma recall is critical
    ]

    rows = []
    for model_name, metrics in all_results.items():
        row = {"model": model_name}
        for m in key_metrics:
            row[m] = metrics.get(m, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("model")

    # Sort by recall (medical priority)
    df = df.sort_values("recall_macro", ascending=False)

    # Save to CSV
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    df.to_csv(csv_path)

    # Save formatted table
    table_path = os.path.join(output_dir, "model_comparison.txt")
    with open(table_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON - Skin Cancer Classification\n")
        f.write("=" * 80 + "\n\n")
        f.write(df.to_string(float_format="%.4f"))
        f.write("\n\n")
        f.write("-" * 80 + "\n")
        f.write("BEST MODEL ANALYSIS (Priority: Recall for Medical Diagnosis)\n")
        f.write("-" * 80 + "\n\n")

        best_recall_model = df["recall_macro"].idxmax()
        best_f1_model = df["f1_macro"].idxmax()
        best_auc_model = df["roc_auc_macro"].idxmax()

        f.write(f"Best Recall (macro):  {best_recall_model} ({df.loc[best_recall_model, 'recall_macro']:.4f})\n")
        f.write(f"Best F1 (macro):      {best_f1_model} ({df.loc[best_f1_model, 'f1_macro']:.4f})\n")
        f.write(f"Best ROC-AUC (macro): {best_auc_model} ({df.loc[best_auc_model, 'roc_auc_macro']:.4f})\n\n")

        f.write(f"RECOMMENDATION: {best_recall_model}\n")
        f.write(f"Justification: In medical imaging, minimizing false negatives (missed diagnoses)\n")
        f.write(f"is critical. The model with the highest recall ensures that potential skin\n")
        f.write(f"cancer cases are not missed, even at the cost of some false positives.\n")
        f.write(f"False positives lead to additional testing; false negatives can be fatal.\n")

    logger.info(f"Saved comparison table to {csv_path} and {table_path}")
    return df


def plot_comparison_bar_chart(
    all_results: Dict[str, Dict[str, float]],
    output_dir: str,
) -> str:
    """Plot bar chart comparing models across key metrics."""
    metrics_to_plot = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    model_names = list(all_results.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, model_name in enumerate(model_names):
        values = [all_results[model_name].get(m, 0) for m in metrics_to_plot]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison - Skin Cancer Classification")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison chart to {path}")
    return path


def evaluate_all_models(
    data_dir: str,
    models_dir: str = "models",
    output_dir: str = "results",
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all trained models and generate comparison report.

    Args:
        data_dir: Path to ISIC dataset
        models_dir: Directory with saved model checkpoints
        output_dir: Directory for evaluation outputs
        batch_size: Batch size for evaluation
        num_workers: Number of data loader workers

    Returns:
        Dictionary mapping model names to their metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Get test loader
    _, _, test_loader, _ = create_data_loaders(
        data_dir, batch_size=batch_size, num_workers=num_workers
    )

    model_names = ["custom_cnn", "resnet50", "efficientnet", "vit"]
    all_results = {}

    for model_name in model_names:
        checkpoint_path = os.path.join(models_dir, f"{model_name}_best.pth")
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping {model_name}")
            continue

        logger.info(f"\nEvaluating {model_name}...")

        # Load model
        model = get_model(model_name)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        # Evaluate
        labels, preds, probs = evaluate_model(model, test_loader, device)

        # Compute metrics
        metrics = compute_metrics(labels, preds, probs)
        all_results[model_name] = metrics

        # Classification report
        report = classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0)
        logger.info(f"\n{model_name} Classification Report:\n{report}")

        report_path = os.path.join(output_dir, f"classification_report_{model_name}.txt")
        with open(report_path, "w") as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("=" * 60 + "\n")
            f.write(report)

        # Plot confusion matrix
        plot_confusion_matrix(labels, preds, model_name, output_dir)

        # Plot ROC curves
        plot_roc_curves(labels, probs, model_name, output_dir)

        # Log to MLflow
        mlflow.set_experiment("skin-cancer-evaluation")
        with mlflow.start_run(run_name=f"eval_{model_name}"):
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

    if all_results:
        # Create comparison table
        comparison_df = create_comparison_table(all_results, output_dir)
        print("\n" + "=" * 60)
        print("MODEL COMPARISON TABLE")
        print("=" * 60)
        print(comparison_df.to_string(float_format="%.4f"))

        # Plot comparison chart
        plot_comparison_bar_chart(all_results, output_dir)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset")
    parser.add_argument("--models-dir", type=str, default="models", help="Saved models dir")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Workers")

    args = parser.parse_args()
    evaluate_all_models(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
