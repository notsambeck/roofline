"""Unit tests for RoofClassifier API."""

import pytest
from PIL import Image
import numpy as np

from roofline import RoofClassifier


class TestRoofClassifier:
    @pytest.fixture
    def classifier(self):
        """Create classifier with untrained model (no weights file)."""
        return RoofClassifier()

    @pytest.fixture
    def dummy_image(self):
        """Create a dummy RGB image."""
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    @pytest.fixture
    def dummy_rgba_image(self):
        """Create a dummy RGBA image (like TIFF with alpha)."""
        arr = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGBA")

    def test_classify_returns_dict(self, classifier, dummy_image):
        """Test classify returns expected dict structure."""
        result = classifier.classify(dummy_image)

        assert isinstance(result, dict)
        assert "class" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_classify_class_is_valid(self, classifier, dummy_image):
        """Test that predicted class is one of the valid classes."""
        result = classifier.classify(dummy_image)

        assert result["class"] in ["flat", "gable", "complex"]

    def test_classify_confidence_range(self, classifier, dummy_image):
        """Test confidence is between 0 and 1."""
        result = classifier.classify(dummy_image)

        assert 0.0 <= result["confidence"] <= 1.0

    def test_classify_probabilities_sum_to_one(self, classifier, dummy_image):
        """Test probabilities sum to approximately 1."""
        result = classifier.classify(dummy_image)

        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.001

    def test_classify_probabilities_all_classes(self, classifier, dummy_image):
        """Test all classes present in probabilities."""
        result = classifier.classify(dummy_image)

        assert set(result["probabilities"].keys()) == {"flat", "gable", "complex"}

    def test_classify_rgba_image(self, classifier, dummy_rgba_image):
        """Test classifier handles RGBA images (converts to RGB)."""
        result = classifier.classify(dummy_rgba_image)

        assert "class" in result
        assert result["class"] in ["flat", "gable", "complex"]

    def test_classify_batch_empty(self, classifier):
        """Test classify_batch with empty list."""
        results = classifier.classify_batch([])
        assert results == []

    def test_classify_batch_single(self, classifier, dummy_image):
        """Test classify_batch with single image."""
        results = classifier.classify_batch([dummy_image])

        assert len(results) == 1
        assert "class" in results[0]

    def test_classify_batch_multiple(self, classifier, dummy_image):
        """Test classify_batch with multiple images."""
        images = [dummy_image, dummy_image, dummy_image]
        results = classifier.classify_batch(images)

        assert len(results) == 3
        for result in results:
            assert "class" in result
            assert "confidence" in result
            assert "probabilities" in result

    def test_classify_different_sizes(self, classifier):
        """Test classifier handles various input sizes (resizes to 224)."""
        sizes = [(100, 100), (224, 224), (512, 512), (300, 200)]

        for w, h in sizes:
            arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            result = classifier.classify(img)
            assert "class" in result
