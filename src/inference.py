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
                checkpoint_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
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
