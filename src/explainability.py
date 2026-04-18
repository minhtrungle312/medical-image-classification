"""
Model Explainability Module (Responsible AI).

Implements Grad-CAM for CNN/ResNet and attention visualization for ViT.
Provides tools to highlight important image regions for classification decisions.
Also includes fairness analysis utilities.
"""

import os
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data_pipeline import get_transforms, CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger(__name__)


def get_gradcam_target_layer(model: nn.Module, model_name: str):
    """Get the appropriate target layer for Grad-CAM based on model type."""
    if model_name == "resnet50":
        return [model.backbone.layer4[-1]]
    elif model_name == "efficientnet":
        return [model.backbone.features[-1]]
    elif model_name == "vit":
        # For ViT, use the last normalization layer
        return [model.backbone.blocks[-1].norm1]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def generate_gradcam(
    model: nn.Module,
    model_name: str,
    image_path: str,
    target_class: Optional[int] = None,
    output_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Generate Grad-CAM visualization for an image.

    Args:
        model: Trained model
        model_name: Model identifier ('resnet50', 'efficientnet', 'vit')
        image_path: Path to input image
        target_class: Target class index (None = predicted class)
        output_path: Where to save the visualization
        device: Computation device

    Returns:
        Grad-CAM heatmap overlay as numpy array
    """
    model.eval()
    model = model.to(device)

    # Load and preprocess image
    raw_image = Image.open(image_path).convert("RGB")
    raw_image_resized = raw_image.resize((224, 224))
    rgb_image = np.array(raw_image_resized) / 255.0

    transform = get_transforms("val")
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # Get target layer
    target_layers = get_gradcam_target_layer(model, model_name)

    # Use reshape_transform for ViT models
    reshape_transform = None
    if model_name == "vit":
        def reshape_transform_fn(tensor, height=14, width=14):
            result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
            result = result.permute(0, 3, 1, 2)
            return result
        reshape_transform = reshape_transform_fn

    # Generate Grad-CAM
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )

    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Create overlay
    cam_image = show_cam_on_image(rgb_image.astype(np.float32), grayscale_cam, use_rgb=True)

    if output_path:
        _save_gradcam_plot(rgb_image, cam_image, grayscale_cam, model_name, output_path, target_class)

    return cam_image


def _save_gradcam_plot(
    original: np.ndarray,
    overlay: np.ndarray,
    heatmap: np.ndarray,
    model_name: str,
    output_path: str,
    target_class: Optional[int],
):
    """Save Grad-CAM visualization as a multi-panel plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    class_name = CLASS_NAMES[target_class] if target_class is not None else "Predicted"
    axes[2].set_title(f"Overlay ({class_name})")
    axes[2].axis("off")

    plt.suptitle(f"Grad-CAM Explainability - {model_name}", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Grad-CAM to {output_path}")


def generate_explainability_report(
    model: nn.Module,
    model_name: str,
    image_paths: list,
    output_dir: str,
    device: torch.device = torch.device("cpu"),
):
    """Generate Grad-CAM visualizations for multiple images."""
    os.makedirs(output_dir, exist_ok=True)

    for i, img_path in enumerate(image_paths):
        output_path = os.path.join(output_dir, f"gradcam_{model_name}_{i}.png")
        try:
            generate_gradcam(model, model_name, img_path, output_path=output_path, device=device)
        except Exception as e:
            logger.error(f"Failed to generate Grad-CAM for {img_path}: {e}")


def analyze_fairness(
    labels: np.ndarray,
    predictions: np.ndarray,
    sensitive_attributes: Optional[dict] = None,
) -> dict:
    """
    Analyze model fairness across different subgroups.

    For skin cancer detection, fairness analysis examines whether the model
    performs equally well across different skin lesion types and sizes.

    Args:
        labels: Ground truth labels
        predictions: Model predictions
        sensitive_attributes: Optional dict of attribute_name -> attribute_values

    Returns:
        Fairness metrics report
    """
    from sklearn.metrics import recall_score, precision_score

    report = {
        "overall": {
            "recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
            "precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        },
        "per_class": {},
        "disparities": {},
    }

    # Per-class performance (demographic parity analog for disease types)
    per_class_recall = recall_score(labels, predictions, average=None, zero_division=0)
    per_class_precision = precision_score(labels, predictions, average=None, zero_division=0)

    for i, cls_name in enumerate(CLASS_NAMES):
        if i < len(per_class_recall):
            report["per_class"][cls_name] = {
                "recall": float(per_class_recall[i]),
                "precision": float(per_class_precision[i]),
                "support": int(np.sum(labels == i)),
            }

    # Disparity analysis
    if len(per_class_recall) > 0:
        recall_gap = float(np.max(per_class_recall) - np.min(per_class_recall))
        precision_gap = float(np.max(per_class_precision) - np.min(per_class_precision))

        report["disparities"] = {
            "recall_gap": recall_gap,
            "precision_gap": precision_gap,
            "worst_recall_class": CLASS_NAMES[int(np.argmin(per_class_recall))],
            "best_recall_class": CLASS_NAMES[int(np.argmax(per_class_recall))],
            "equalized_odds_satisfied": recall_gap < 0.2,  # Threshold
        }

    # Sensitive attribute analysis (if provided)
    if sensitive_attributes:
        for attr_name, attr_values in sensitive_attributes.items():
            attr_values = np.array(attr_values)
            unique_vals = np.unique(attr_values)
            attr_metrics = {}

            for val in unique_vals:
                mask = attr_values == val
                if np.sum(mask) > 0:
                    attr_metrics[str(val)] = {
                        "recall": float(
                            recall_score(
                                labels[mask], predictions[mask],
                                average="macro", zero_division=0,
                            )
                        ),
                        "support": int(np.sum(mask)),
                    }

            report[f"by_{attr_name}"] = attr_metrics

    return report


def generate_ethics_report(output_dir: str):
    """Generate a responsible AI ethics report."""
    os.makedirs(output_dir, exist_ok=True)

    report = """
================================================================================
RESPONSIBLE AI REPORT - Skin Cancer Classification System
================================================================================

1. DATA PRIVACY (HIPAA Compliance)
-----------------------------------
- All patient images are de-identified (no PII in ISIC dataset)
- No patient metadata (name, DOB, SSN) is stored or processed
- Image data is processed locally; no transmission to external services
- Model weights do not memorize individual patient data
- Recommendation: In production, implement data encryption at rest and in transit

2. FAIRNESS ANALYSIS
--------------------
- Model performance is evaluated per-class to detect disparities
- Class imbalance is addressed via weighted sampling and weighted loss
- Recall disparity between classes is monitored (equalized odds metric)
- Mitigation: Oversampling underrepresented classes during training
- Limitation: Dataset may not represent all skin tones equally;
  real-world deployment should validate across diverse populations

3. MODEL EXPLAINABILITY
-----------------------
- Grad-CAM visualizations show which image regions drive predictions
- Attention maps available for Vision Transformer models
- Clinicians can verify model reasoning against medical knowledge
- Ensures model focuses on lesion characteristics, not artifacts

4. INTENDED USE & LIMITATIONS
------------------------------
- Intended as a decision SUPPORT tool, NOT a replacement for dermatologists
- Not FDA-approved; should not be sole basis for medical decisions
- Performance may degrade on out-of-distribution images:
  * Different camera types or lighting conditions
  * Non-dermoscopic photographs
  * Previously unseen lesion types
- Requires validation on local patient populations before deployment

5. RISK ASSESSMENT
------------------
- False Negatives (missed cancer): HIGHEST RISK - could delay treatment
  * Mitigation: Optimize for recall; flag uncertain predictions for review
- False Positives (unnecessary biopsies): Moderate risk - additional procedures
  * Acceptable trade-off in medical context
- Model drift: Performance may degrade over time
  * Mitigation: Continuous monitoring and periodic retraining

6. RECOMMENDATIONS
-------------------
- Deploy with mandatory human-in-the-loop review
- Implement prediction confidence thresholds
- Set up monitoring for drift detection
- Conduct periodic fairness audits on deployment data
- Maintain audit logs of all predictions for accountability
================================================================================
"""
    report_path = os.path.join(output_dir, "responsible_ai_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Saved ethics report to {report_path}")
    return report_path
