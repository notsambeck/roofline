"""RoofNet CNN architecture for roof type classification."""

import torch
import torch.nn as nn
from torchvision import models


class RoofNet(nn.Module):
    """ResNet18-based classifier for roof types.

    Uses pretrained ResNet18 backbone with custom classifier head.
    Backbone is frozen by default for efficient training.

    Input: (batch, 3, 224, 224) RGB images
    Output: (batch, 4) logits for [flat, gable, complex, bug]
    """

    CLASSES = ["flat", "gable", "complex", "bug"]
    NUM_CLASSES = 4
    INPUT_SIZE = 224

    def __init__(self, freeze_backbone: bool = True):
        """Initialize model.

        Args:
            freeze_backbone: If True, freeze ResNet weights and only train classifier.
        """
        super().__init__()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classifier head
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, self.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
