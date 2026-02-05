#!/usr/bin/env python3
"""Show examples of augmentations."""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from roofline.data import RandomShadow, RandomTreeOcclusion

def main():
    data_dir = Path(__file__).parent.parent / "data"

    # Get a few sample images
    images = list((data_dir / "Gable_hip_other").glob("*.tif"))[:4]
    images += list((data_dir / "Flat_data").glob("*.tif"))[:4]
    random.shuffle(images)
    images = images[:4]

    shadow = RandomShadow(p=1.0, max_coverage=0.4)
    tree = RandomTreeOcclusion(p=1.0, max_coverage=0.9)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for row, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB").resize((224, 224))

        # Original
        axes[row, 0].imshow(img)
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis("off")

        # Shadow only
        shadow_img = shadow(img.copy())
        axes[row, 1].imshow(shadow_img)
        axes[row, 1].set_title("Shadow (40%)" if row == 0 else "")
        axes[row, 1].axis("off")

        # Tree only
        tree_img = tree(img.copy())
        axes[row, 2].imshow(tree_img)
        axes[row, 2].set_title("Tree (90%)" if row == 0 else "")
        axes[row, 2].axis("off")

        # Both
        both_img = tree(shadow(img.copy()))
        axes[row, 3].imshow(both_img)
        axes[row, 3].set_title("Both" if row == 0 else "")
        axes[row, 3].axis("off")

    plt.suptitle("Occlusion Augmentations", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = Path(__file__).parent.parent / "augmentation_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
