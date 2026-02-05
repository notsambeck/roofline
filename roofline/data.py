"""Dataset utilities for roof type classification."""

import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms


class RandomDownscale:
    """Random downscale then upscale (simulates lower resolution source)."""

    def __init__(self, p: float = 0.5, min_scale: float = 0.3, max_scale: float = 1.0):
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        scale = random.uniform(self.min_scale, self.max_scale)

        # Downscale
        small_w = max(1, int(w * scale))
        small_h = max(1, int(h * scale))
        small = img.resize((small_w, small_h), Image.BILINEAR)

        # Upscale back to original
        return small.resize((w, h), Image.BILINEAR)


class RandomBlur:
    """Random Gaussian blur."""

    def __init__(self, p: float = 0.5, max_radius: float = 10.0):
        self.p = p
        self.max_radius = max_radius

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        from PIL import ImageFilter
        radius = random.uniform(0.5, self.max_radius)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomZoom:
    """Random zoom in/out augmentation.

    Zooms between min_scale and max_scale, padding or cropping as needed.
    """

    def __init__(self, p: float = 0.3, min_scale: float = 0.6, max_scale: float = 1.4):
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        scale = random.uniform(self.min_scale, self.max_scale)

        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = img.resize((new_w, new_h), Image.BILINEAR)

        if scale > 1.0:
            # Zoom in: crop center
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            result = resized.crop((left, top, left + w, top + h))
        else:
            # Zoom out: pad with edge pixels
            result = Image.new("RGB", (w, h))
            # Fill with average edge color
            avg_color = tuple(int(c) for c in np.array(img).mean(axis=(0, 1)))
            result.paste(avg_color, (0, 0, w, h))
            # Paste resized in center
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            result.paste(resized, (left, top))

        return result


class RandomShadow:
    """Apply random shadow (darkened polygon) to image.

    Simulates shadows from buildings, clouds, etc.
    """

    def __init__(self, p: float = 0.3, max_coverage: float = 0.4, darkness: tuple[float, float] = (0.3, 0.7)):
        """
        Args:
            p: Probability of applying shadow
            max_coverage: Maximum fraction of image to cover (0-1)
            darkness: Range of darkness multiplier (0=black, 1=no change)
        """
        self.p = p
        self.max_coverage = max_coverage
        self.darkness = darkness

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        img = img.copy()

        # Create shadow mask
        mask = Image.new("L", (w, h), 255)
        draw = ImageDraw.Draw(mask)

        # Random coverage for this shadow
        coverage = random.uniform(0.05, self.max_coverage)
        target_area = w * h * coverage

        # Generate random polygon points
        n_points = random.randint(4, 8)

        # Start from a random edge or corner
        cx = random.randint(0, w)
        cy = random.randint(0, h)

        points = []
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points + random.uniform(-0.5, 0.5)
            radius = np.sqrt(target_area / np.pi) * random.uniform(0.5, 1.5)
            px = cx + radius * np.cos(angle)
            py = cy + radius * np.sin(angle)
            points.append((px, py))

        # Draw shadow polygon
        darkness = int(255 * random.uniform(*self.darkness))
        draw.polygon(points, fill=darkness)

        # Apply mask to darken image
        img_array = np.array(img).astype(np.float32)
        mask_array = np.array(mask).astype(np.float32) / 255.0
        mask_array = mask_array[:, :, np.newaxis]

        result = (img_array * mask_array).astype(np.uint8)
        return Image.fromarray(result)


class RandomTreeOcclusion:
    """Add random tree-like occlusion (green/dark irregular blobs).

    Simulates tree canopy covering parts of the roof.
    """

    def __init__(self, p: float = 0.3, max_coverage: float = 0.5, n_blobs: tuple[int, int] = (1, 4)):
        """
        Args:
            p: Probability of applying tree occlusion
            max_coverage: Maximum fraction of image to cover (0-1)
            n_blobs: Range for number of tree blobs to add
        """
        self.p = p
        self.max_coverage = max_coverage
        self.n_blobs = n_blobs

        # Tree-like colors (dark greens, some brown)
        self.colors = [
            (34, 85, 34),    # Dark green
            (45, 90, 39),    # Forest green
            (55, 100, 45),   # Medium green
            (30, 70, 30),    # Deep green
            (60, 80, 40),    # Olive green
            (40, 60, 35),    # Dark olive
        ]

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        img = img.copy()
        draw = ImageDraw.Draw(img, "RGBA")

        n_blobs = random.randint(*self.n_blobs)
        coverage_per_blob = self.max_coverage / n_blobs

        for _ in range(n_blobs):
            # Random center for this blob
            cx = random.randint(0, w)
            cy = random.randint(0, h)

            # Random size based on coverage
            target_area = w * h * random.uniform(0.02, coverage_per_blob)
            base_radius = np.sqrt(target_area / np.pi)

            # Create irregular blob with multiple overlapping ellipses
            color = random.choice(self.colors)
            alpha = random.randint(180, 240)

            n_ellipses = random.randint(3, 7)
            for _ in range(n_ellipses):
                # Offset from center
                ox = cx + random.gauss(0, base_radius * 0.3)
                oy = cy + random.gauss(0, base_radius * 0.3)

                # Random ellipse size
                rx = base_radius * random.uniform(0.3, 0.8)
                ry = base_radius * random.uniform(0.3, 0.8)

                # Slight color variation
                c = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in color)

                bbox = (ox - rx, oy - ry, ox + rx, oy + ry)
                draw.ellipse(bbox, fill=(*c, alpha))

        return img.convert("RGB")


# Mapping from directory names to class labels
FOLDER_TO_CLASS = {
    "Flat_data": "flat",
    "flat": "flat",
    "Gable_hip_other": "gable",
    "gable": "gable",
    "Complex_data": "complex",
    "complex": "complex",
}


def get_train_transforms() -> transforms.Compose:
    """Get training data augmentation transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.8, saturation=1.0, hue=0.5),
        transforms.RandomGrayscale(p=0.3),
        RandomZoom(p=0.3, min_scale=0.6, max_scale=1.4),
        RandomBlur(p=0.5, max_radius=10.0),
        RandomDownscale(p=0.5, min_scale=0.3, max_scale=1.0),
        RandomShadow(p=0.3, max_coverage=0.4),
        RandomTreeOcclusion(p=0.5, max_coverage=0.9),
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

    CLASSES = ["flat", "gable", "complex"]

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
