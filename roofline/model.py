"""RoofNet CNN architecture for roof type classification."""

import torch
import torch.nn as nn


class RoofNet(nn.Module):
    """Simple CNN for classifying roof types.

    Architecture: 4 conv blocks with BatchNorm + classifier head.
    Input: (batch, 3, 224, 224) RGB images
    Output: (batch, 4) logits for [flat, gable, complex, bug]
    """

    CLASSES = ["flat", "gable", "complex", "bug"]
    NUM_CLASSES = 4
    INPUT_SIZE = 224

    def __init__(self):
        super().__init__()

        # Conv block 1: 3 -> 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Conv block 2: 32 -> 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Conv block 3: 64 -> 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Conv block 4: 128 -> 256 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Global average pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224)

        Returns:
            Logits tensor of shape (batch, 4)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
