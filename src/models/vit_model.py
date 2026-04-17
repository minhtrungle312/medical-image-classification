"""
Vision Transformer (ViT) model for skin cancer classification.

Uses a pretrained ViT from the timm library, fine-tuned for
dermoscopic image classification.
"""

import torch
import torch.nn as nn
import timm


class ViTModel(nn.Module):
    """
    Vision Transformer for skin cancer classification.

    Uses ViT-Base/16 pretrained on ImageNet, with a custom classification head.
    """

    def __init__(
        self,
        num_classes: int = 9,
        model_name: str = "vit_base_patch16_224",
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # Load pretrained ViT from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
        )

        if freeze_backbone:
            # Freeze all transformer blocks except the last 2
            for name, param in self.backbone.named_parameters():
                if "blocks.11" not in name and "blocks.10" not in name and "norm" not in name:
                    param.requires_grad = False

        # Get feature dimension
        feature_dim = self.backbone.num_features

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_attention_maps(self, x: torch.Tensor):
        """Extract attention maps from the last transformer block (for explainability)."""
        self.backbone.eval()
        attention_maps = []

        def hook_fn(module, input, output):
            # output shape: (batch, num_heads, seq_len, seq_len)
            attention_maps.append(output.detach())

        # Register hook on the last attention block
        hook = self.backbone.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = self.backbone(x)

        hook.remove()
        return attention_maps[0] if attention_maps else None
