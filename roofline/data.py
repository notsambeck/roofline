"""Dataset utilities for roof type classification."""

import os
from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Mapping from directory names to class labels
FOLDER_TO_CLASS = {
    "Flat_data": "flat",
    "flat": "flat",
    "Gable_hip_other": "gable",
    "gable": "gable",
    "Complex_data": "complex",
    "complex": "complex",
    "Bugs": "bug",
    "bug": "bug",
}


def get_train_transforms() -> transforms.Compose:
    """Get training data augmentation transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])


def get_val_transforms() -> transforms.Compose:
    """Get validation/inference transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class RoofDataset(Dataset):
    """Dataset for loading roof images from folder structure.

    Expected structure:
        data_dir/
            flat/ or Flat_data/
            gable/ or Gable_hip_other/
            complex/ or Complex_data/
            bug/ or Bugs/
    """

    CLASSES = ["flat", "gable", "complex", "bug"]

    def __init__(
        self,
        data_dir: str | Path,
        transform: Optional[transforms.Compose] = None,
        train: bool = True,
    ):
        """Initialize dataset.

        Args:
            data_dir: Root directory containing class folders
            transform: Optional transforms to apply. If None, uses default train/val transforms.
            train: If True and transform is None, use training augmentations.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform or (get_train_transforms() if train else get_val_transforms())

        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        """Scan data directory and load image paths."""
        for folder_name, class_name in FOLDER_TO_CLASS.items():
            folder_path = self.data_dir / folder_name
            if not folder_path.exists():
                continue

            class_idx = self.CLASSES.index(class_name)

            for img_path in folder_path.iterdir():
                if img_path.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Get a sample.

        Returns:
            tuple: (image_tensor, class_index)
        """
        img_path, class_idx = self.samples[idx]

        # Load image and convert to RGB (handles RGBA TIFFs)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, class_idx

    @classmethod
    def class_name(cls, idx: int) -> str:
        """Get class name from index."""
        return cls.CLASSES[idx]

    @classmethod
    def class_index(cls, name: str) -> int:
        """Get class index from name."""
        return cls.CLASSES.index(name)
