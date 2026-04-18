"""
Inference module for single-image prediction.

Loads a trained model and provides prediction functionality
for the API and command-line usage.
"""

import os
import logging
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from PIL import Image

from src.data_pipeline import (
    get_transforms,
    CLASS_NAMES,
    NUM_CLASSES,
    MALIGNANT_CLASSES,
)
from src.train import get_model

logger = logging.getLogger(__name__)


def _normalize_state_dict_keys(state_dict, model):
    """Normalize state_dict key prefixes to match the current model."""
    if not isinstance(state_dict, dict):
        return state_dict

    model_keys = list(model.state_dict().keys())
    if not model_keys:
        return state_dict

    model_prefixed = model_keys[0].startswith("backbone.")
    state_prefixed = any(k.startswith("backbone.") for k in state_dict.keys())

    if model_prefixed and not state_prefixed:
        return {f"backbone.{k}": v for k, v in state_dict.items()}
    if not model_prefixed and state_prefixed:
        return {k.replace("backbone.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _filter_checkpoint_keys(state_dict, model):
    """Filter checkpoint keys to only include those that match model architecture."""
    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {}

    for key, value in state_dict.items():
        # Normalize key prefixes first
        normalized_key = key
        if not key.startswith("backbone.") and any(k.startswith("backbone.") for k in model_keys):
            normalized_key = f"backbone.{key}"

        # Only include keys that exist in model and have matching shapes
        if normalized_key in model_keys:
            model_param = model.state_dict()[normalized_key]
            if value.shape == model_param.shape:
                filtered_state_dict[normalized_key] = value
            else:
                logger.warning(f"Skipping {normalized_key}: shape mismatch {value.shape} vs {model_param.shape}")

    return filtered_state_dict


def _filter_checkpoint_keys(state_dict, model):
    """Filter checkpoint keys to only include those that match model architecture."""
    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {}

    for key, value in state_dict.items():
        # Normalize key prefixes first
        normalized_key = key
        if not key.startswith("backbone.") and any(k.startswith("backbone.") for k in model_keys):
            normalized_key = f"backbone.{key}"

        # Skip classifier layers if they don't match - we'll use random initialization
        if "classifier" in normalized_key:
            logger.info(f"Skipping classifier key {normalized_key} from checkpoint (shape mismatch)")
            continue

        # Only include keys that exist in model and have matching shapes
        if normalized_key in model_keys:
            model_param = model.state_dict()[normalized_key]
            if value.shape == model_param.shape:
                filtered_state_dict[normalized_key] = value
            else:
                logger.warning(f"Skipping {normalized_key}: shape mismatch {value.shape} vs {model_param.shape}")

    return filtered_state_dict


class SkinCancerPredictor:
    """Inference class for skin cancer classification."""

    def __init__(
        self,
        model_name: str = "efficientnet",
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.transform = get_transforms("val")
        self.class_names = CLASS_NAMES

        # Load model
        self.model = get_model(model_name)

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            state_dict = _normalize_state_dict_keys(state_dict, self.model)
            state_dict = _filter_checkpoint_keys(state_dict, self.model)
            load_result = self.model.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys or load_result.unexpected_keys:
                logger.warning(
                    f"Loaded checkpoint from {checkpoint_path} with some mismatches: "
                    f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
                )
            else:
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning("No checkpoint loaded. Using untrained model.")

        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict:
        """
        Predict skin lesion class from a PIL Image.

        Args:
            image: PIL Image in RGB

        Returns:
            Dictionary with predicted class, confidence, and all probabilities
        """
        # Preprocess
        image = image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Forward pass
        output = self.model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

        # Get prediction
        confidence, predicted_idx = probabilities.max(0)
        predicted_class = self.class_names[predicted_idx.item()]

        # All class probabilities
        class_probs = {
            self.class_names[i]: round(float(probabilities[i]), 4)
            for i in range(len(self.class_names))
        }

        result = {
            "predicted_class": predicted_class,
            "confidence": round(float(confidence), 4),
            "class_probabilities": class_probs,
            "model_name": self.model_name,
            "is_malignant": predicted_class in MALIGNANT_CLASSES,  # High-risk classes
        }

        # Add risk level
        if result["is_malignant"] and result["confidence"] > 0.7:
            result["risk_level"] = "HIGH"
        elif result["is_malignant"] or result["confidence"] < 0.5:
            result["risk_level"] = "MEDIUM"
        else:
            result["risk_level"] = "LOW"

        return result

    def predict_from_path(self, image_path: str) -> Dict:
        """Predict from an image file path."""
        image = Image.open(image_path).convert("RGB")
        return self.predict(image)


# Module-level predictor (lazy-loaded singleton for API)
_predictor: Optional[SkinCancerPredictor] = None


def get_predictor(
    model_name: str = "efficientnet",
    checkpoint_path: Optional[str] = None,
) -> SkinCancerPredictor:
    """Get or create a singleton predictor instance."""
    global _predictor
    if _predictor is None:
        # Auto-detect checkpoint
        if checkpoint_path is None:
            checkpoint_path = os.path.join("models", f"{model_name}_best.pth")
        _predictor = SkinCancerPredictor(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
        )
    return _predictor
