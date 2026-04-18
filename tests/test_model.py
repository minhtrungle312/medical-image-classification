"""
Unit tests for model architectures.

Validates that all models produce correct output shapes,
handle different batch sizes, and have expected properties.
"""

import pytest
import torch

from src.models.transfer_learning import ResNet50Model, EfficientNetModel
from src.models.vit_model import ViTModel
from src.data_pipeline import NUM_CLASSES


@pytest.fixture
def sample_input():
    """Create a sample input tensor (batch=2, 3 channels, 224x224)."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def single_input():
    """Create a single-image input tensor."""
    return torch.randn(1, 3, 224, 224)


# --- ResNet50 Tests ---


class TestResNet50:
    def test_output_shape(self, sample_input):
        model = ResNet50Model(num_classes=NUM_CLASSES)
        output = model(sample_input)
        assert output.shape == (2, NUM_CLASSES)

    def test_frozen_layers(self):
        model = ResNet50Model(num_classes=NUM_CLASSES, freeze_backbone=True)
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        total = sum(1 for p in model.parameters())
        assert frozen > 0
        assert frozen < total

    def test_unfreeze(self):
        model = ResNet50Model(num_classes=NUM_CLASSES, freeze_backbone=True)
        model.unfreeze_all()
        unfrozen = sum(1 for p in model.parameters() if p.requires_grad)
        total = sum(1 for p in model.parameters())
        assert unfrozen == total

    def test_target_layer(self):
        model = ResNet50Model(num_classes=NUM_CLASSES)
        target = model.get_target_layer()
        assert target is not None


# --- EfficientNet Tests ---


class TestEfficientNet:
    def test_output_shape(self, sample_input):
        model = EfficientNetModel(num_classes=NUM_CLASSES)
        output = model(sample_input)
        assert output.shape == (2, NUM_CLASSES)

    def test_frozen_layers(self):
        model = EfficientNetModel(num_classes=NUM_CLASSES, freeze_backbone=True)
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        assert frozen > 0

    def test_target_layer(self):
        model = EfficientNetModel(num_classes=NUM_CLASSES)
        target = model.get_target_layer()
        assert target is not None


# --- ViT Tests ---


class TestViT:
    def test_output_shape(self, sample_input):
        model = ViTModel(num_classes=NUM_CLASSES)
        output = model(sample_input)
        assert output.shape == (2, NUM_CLASSES)

    def test_frozen_layers(self):
        model = ViTModel(num_classes=NUM_CLASSES, freeze_backbone=True)
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        assert frozen > 0

    def test_unfreeze(self):
        model = ViTModel(num_classes=NUM_CLASSES, freeze_backbone=True)
        model.unfreeze_all()
        unfrozen = sum(1 for p in model.parameters() if p.requires_grad)
        total = sum(1 for p in model.parameters())
        assert unfrozen == total


# --- Model Factory Tests ---


class TestModelFactory:
    def test_get_all_models(self, sample_input):
        from src.train import get_model

        for name in ["resnet50", "efficientnet", "vit"]:
            model = get_model(name)
            output = model(sample_input)
            assert output.shape == (2, NUM_CLASSES), f"Failed for {name}"

    def test_invalid_model_name(self):
        from src.train import get_model

        with pytest.raises(ValueError):
            get_model("invalid_model")
