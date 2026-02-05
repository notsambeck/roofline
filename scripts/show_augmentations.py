#!/usr/bin/env python3
"""Show examples of augmentations."""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from roofline.data import RandomShadow, RandomTreeOcclusion, RandomZoom, RandomBlur, RandomDownscale

def main():
    data_dir = Path(__file__).parent.parent / "data"

    # Get a few sample images
    images = list((data_dir / "Gable_hip_other").glob("*.tif"))[:4]
    images += list((data_dir / "Flat_data").glob("*.tif"))[:4]
    random.shuffle(images)
    images = images[:4]

    # Augmentations
    rotation = transforms.RandomRotation(90)
    color_jitter = transforms.ColorJitter(contrast=0.8, saturation=1.0, hue=0.5)
    grayscale = transforms.Grayscale(num_output_channels=3)
    zoom = RandomZoom(p=1.0, min_scale=0.6, max_scale=1.4)
    blur = RandomBlur(p=1.0, max_radius=10.0)
    downscale = RandomDownscale(p=1.0, min_scale=0.3, max_scale=0.3)  # Show worst case
    shadow = RandomShadow(p=1.0, max_coverage=0.4)
    tree = RandomTreeOcclusion(p=1.0, max_coverage=0.9)

    fig, axes = plt.subplots(4, 9, figsize=(27, 12))

    for row, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB").resize((224, 224))

        # Original
        axes[row, 0].imshow(img)
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis("off")

        # Rotation
        rot_img = rotation(img.copy())
        axes[row, 1].imshow(rot_img)
        axes[row, 1].set_title("Rotation 90°" if row == 0 else "")
        axes[row, 1].axis("off")

        # Color jitter
        color_img = color_jitter(img.copy())
        axes[row, 2].imshow(color_img)
        axes[row, 2].set_title("Color Jitter" if row == 0 else "")
        axes[row, 2].axis("off")

        # Grayscale
        gray_img = grayscale(img.copy())
        axes[row, 3].imshow(gray_img)
        axes[row, 3].set_title("Grayscale" if row == 0 else "")
        axes[row, 3].axis("off")

        # Zoom
        zoom_img = zoom(img.copy())
        axes[row, 4].imshow(zoom_img)
        axes[row, 4].set_title("Zoom 60-140%" if row == 0 else "")
        axes[row, 4].axis("off")

        # Blur
        blur_img = blur(img.copy())
        axes[row, 5].imshow(blur_img)
        axes[row, 5].set_title("Blur 10px" if row == 0 else "")
        axes[row, 5].axis("off")

        # Downscale
        down_img = downscale(img.copy())
        axes[row, 6].imshow(down_img)
        axes[row, 6].set_title("Downscale 30%" if row == 0 else "")
        axes[row, 6].axis("off")

        # Shadow + Tree
        occl_img = tree(shadow(img.copy()))
        axes[row, 7].imshow(occl_img)
        axes[row, 7].set_title("Shadow+Tree" if row == 0 else "")
        axes[row, 7].axis("off")

        # All combined
        all_img = tree(shadow(downscale(blur(zoom(grayscale(color_jitter(rotation(img.copy()))))))))
        axes[row, 8].imshow(all_img)
        axes[row, 8].set_title("All Combined" if row == 0 else "")
        axes[row, 8].axis("off")

    plt.suptitle("Data Augmentations", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = Path(__file__).parent.parent / "augmentation_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
