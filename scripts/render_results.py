#!/usr/bin/env python3
"""Render test results as a 5x5 graphic."""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from roofline import RoofClassifier
from roofline.data import FOLDER_TO_CLASS


def get_sample_images(data_dir: Path, n_per_class: int = 6) -> list[tuple[Path, str]]:
    """Get random sample images from each class."""
    samples = []
    seen_classes = set()

    for folder_name, class_name in FOLDER_TO_CLASS.items():
        if class_name in seen_classes:
            continue
        seen_classes.add(class_name)

        folder_path = data_dir / folder_name
        if not folder_path.exists():
            continue

        images = list(folder_path.glob("*.tif")) + list(folder_path.glob("*.tiff"))
        if images:
            selected = random.sample(images, min(n_per_class, len(images)))
            for img_path in selected:
                samples.append((img_path, class_name))

    random.shuffle(samples)
    return samples[:25]  # Limit to 25 for 5x5 grid


def main():
    data_dir = Path(__file__).parent.parent / "data"
    classifier = RoofClassifier()

    samples = get_sample_images(data_dir, n_per_class=7)

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.flatten()

    correct = 0
    for i, ax in enumerate(axes):
        if i >= len(samples):
            ax.axis("off")
            continue

        img_path, true_class = samples[i]
        img = Image.open(img_path).convert("RGB")
        result = classifier.classify(img)
        pred_class = result["class"]
        conf = result["confidence"]

        ax.imshow(img)
        ax.axis("off")

        is_correct = pred_class == true_class
        if is_correct:
            correct += 1
        color = "green" if is_correct else "red"
        symbol = "✓" if is_correct else "✗"

        ax.set_title(
            f"{symbol} True: {true_class}\nPred: {pred_class} ({conf:.0%})",
            fontsize=10,
            color=color,
            fontweight="bold"
        )

    acc = correct / len(samples) * 100
    plt.suptitle(f"Roof Classification Results ({correct}/{len(samples)} = {acc:.0f}%)",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = Path(__file__).parent.parent / "test_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
