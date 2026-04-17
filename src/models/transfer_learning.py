"""
Transfer Learning models for skin cancer classification.

Implements fine-tuned ResNet50 and EfficientNet-B0 with pretrained
ImageNet weights. Only the last layers are fine-tuned.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet50Model(nn.Module):
    """
    ResNet50 with transfer learning.

    Freezes early layers and fine-tunes the last residual block + classifier.
    """

    def __init__(self, num_classes: int = 9, freeze_backbone: bool = False):
        super().__init__()

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        if freeze_backbone:
            # Freeze all layers except layer4 and fc
            for name, param in self.backbone.named_parameters():
                if "layer4" not in name and "fc" not in name:
                    param.requires_grad = False

        # Replace the final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_target_layer(self):
        """Return the target layer for Grad-CAM."""
        return self.backbone.layer4[-1]


class EfficientNetModel(nn.Module):
    """
    EfficientNet-B0 with transfer learning.

    Freezes the feature extractor and fine-tunes the classifier.
    """

    def __init__(self, num_classes: int = 9, freeze_backbone: bool = False):
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        if freeze_backbone:
            # Freeze all feature layers
            for name, param in self.backbone.named_parameters():
                if "classifier" not in name and "features.8" not in name:
                    param.requires_grad = False

        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_target_layer(self):
        """Return the target layer for Grad-CAM."""
        return self.backbone.features[-1]
