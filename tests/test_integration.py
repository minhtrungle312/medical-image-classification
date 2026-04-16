"""
Integration tests for the prediction pipeline.

Tests end-to-end flow from image input to prediction output,
including model loading, inference, and API integration.
"""

import io
import pytest
import torch
from PIL import Image
from unittest.mock import MagicMock

from src.models.custom_cnn import CustomCNN
from src.models.transfer_learning import ResNet50Model, EfficientNetModel
from src.models.vit_model import ViTModel
from src.data_pipeline import NUM_CLASSES, get_transforms
from src.inference import SkinCancerPredictor


class TestEndToEndInference:
    """Test full inference pipeline (model -> prediction)."""

    @pytest.fixture
    def sample_image(self):
        return Image.new("RGB", (224, 224), color=(128, 100, 80))

    def test_custom_cnn_inference(self, sample_image):
        predictor = SkinCancerPredictor(model_name="custom_cnn")
        result = predictor.predict(sample_image)

        assert "predicted_class" in result
        assert "confidence" in result
        assert "class_probabilities" in result
        assert "risk_level" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["class_probabilities"]) == NUM_CLASSES

    def test_resnet50_inference(self, sample_image):
        predictor = SkinCancerPredictor(model_name="resnet50")
        result = predictor.predict(sample_image)
        assert result["predicted_class"] in [
            "actinic keratosis", "basal cell carcinoma", "dermatofibroma",
            "melanoma", "nevus", "pigmented benign keratosis",
            "seborrheic keratosis", "squamous cell carcinoma", "vascular lesion",
        ]

    def test_efficientnet_inference(self, sample_image):
        predictor = SkinCancerPredictor(model_name="efficientnet")
        result = predictor.predict(sample_image)
        assert isinstance(result["is_malignant"], bool)

    def test_vit_inference(self, sample_image):
        predictor = SkinCancerPredictor(model_name="vit")
        result = predictor.predict(sample_image)
        assert sum(result["class_probabilities"].values()) == pytest.approx(1.0, abs=0.01)

    def test_probabilities_sum_to_one(self, sample_image):
        predictor = SkinCancerPredictor(model_name="custom_cnn")
        result = predictor.predict(sample_image)
        total_prob = sum(result["class_probabilities"].values())
        assert total_prob == pytest.approx(1.0, abs=0.01)

    def test_risk_level_logic(self, sample_image):
        predictor = SkinCancerPredictor(model_name="custom_cnn")
        result = predictor.predict(sample_image)
        assert result["risk_level"] in ["HIGH", "MEDIUM", "LOW"]


class TestModelValidation:
    """Model validation tests - ensure models meet minimum quality criteria."""

    @pytest.fixture
    def batch_inputs(self):
        return torch.randn(4, 3, 224, 224)

    def test_output_probabilities_valid(self, batch_inputs):
        """Outputs should produce valid probability distributions."""
        model = CustomCNN(num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            outputs = model(batch_inputs)
            probs = torch.softmax(outputs, dim=1)
            # Each row should sum to 1
            sums = probs.sum(dim=1)
            assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_model_deterministic_in_eval(self):
        """Model should produce consistent outputs in eval mode."""
        model = CustomCNN(num_classes=NUM_CLASSES)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_model_handles_different_batch_sizes(self):
        """Model should work with different batch sizes."""
        model = CustomCNN(num_classes=NUM_CLASSES)
        model.eval()
        for bs in [1, 2, 8]:
            x = torch.randn(bs, 3, 224, 224)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (bs, NUM_CLASSES)

    def test_gradient_flow(self):
        """Gradients should flow through all trainable parameters."""
        model = CustomCNN(num_classes=NUM_CLASSES)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"
