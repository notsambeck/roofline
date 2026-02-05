"""High-level inference API for roof classification."""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .data import get_val_transforms
from .model import RoofNet


class RoofClassifier:
    """High-level API for classifying roof types from images.

    Example:
        classifier = RoofClassifier()
        img = Image.open("roof.tif")
        result = classifier.classify(img)
        # {"class": "gable", "confidence": 0.92, "probabilities": {...}}
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        device: Optional[str] = None,
    ):
        """Initialize the classifier.

        Args:
            model_path: Path to trained model weights. If None, looks for
                weights/model.pt relative to the package.
            device: Device to run inference on. Auto-detected if None.
        """
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Load model (ResNet18 backbone, no need to freeze for inference)
        self.model = RoofNet(freeze_backbone=False)

        if model_path is None:
            # Default to weights/model.pt, fall back to distributable weights
            model_path = Path(__file__).parent.parent / "weights" / "model.pt"
            if not model_path.exists():
                model_path = model_path.with_suffix(".pt.dist")

        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # Setup transforms
        self.transform = get_val_transforms()

    def classify(self, image: Image.Image) -> dict:
        """Classify a single image.

        Args:
            image: PIL Image (any mode, will be converted to RGB)

        Returns:
            dict with keys:
                - "class": predicted class name
                - "confidence": confidence score (0-1)
                - "probabilities": dict of class -> probability
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Transform and add batch dimension
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        # Get prediction
        confidence, pred_idx = probs.max(0)
        pred_class = RoofNet.CLASSES[pred_idx.item()]

        # Build probabilities dict
        probabilities = {
            cls: probs[i].item() for i, cls in enumerate(RoofNet.CLASSES)
        }

        return {
            "class": pred_class,
            "confidence": confidence.item(),
            "probabilities": probabilities,
        }

    def classify_batch(self, images: list[Image.Image]) -> list[dict]:
        """Classify a batch of images.

        Args:
            images: List of PIL Images

        Returns:
            List of result dicts (same format as classify())
        """
        if not images:
            return []

        # Convert and transform all images
        tensors = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensors.append(self.transform(img))

        # Stack into batch
        batch = torch.stack(tensors).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)

        # Build results
        results = []
        for i in range(len(images)):
            confidence, pred_idx = probs[i].max(0)
            pred_class = RoofNet.CLASSES[pred_idx.item()]

            probabilities = {
                cls: probs[i][j].item() for j, cls in enumerate(RoofNet.CLASSES)
            }

            results.append({
                "class": pred_class,
                "confidence": confidence.item(),
                "probabilities": probabilities,
            })

        return results
