"""Unit tests for RoofNet model."""

import torch
import pytest

from roofline.model import RoofNet


class TestRoofNet:
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        model = RoofNet()
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        output = model(x)

        assert output.shape == (batch_size, 4)

    def test_single_image(self):
        """Test forward pass with single image."""
        model = RoofNet()
        x = torch.randn(1, 3, 224, 224)
        output = model(x)

        assert output.shape == (1, 4)

    def test_classes(self):
        """Test class constants."""
        assert RoofNet.CLASSES == ["flat", "gable", "complex", "bug"]
        assert RoofNet.NUM_CLASSES == 4
        assert RoofNet.INPUT_SIZE == 224

    def test_parameter_count(self):
        """Test parameter count is reasonable (~400K)."""
        model = RoofNet()
        params = model.count_parameters()

        # Should be around 400K, allow some variance
        assert 300_000 < params < 600_000

    def test_eval_mode(self):
        """Test model works in eval mode (BatchNorm behavior)."""
        model = RoofNet()
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 4)

    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        model = RoofNet()
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape
