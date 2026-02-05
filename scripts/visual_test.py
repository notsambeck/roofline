#!/usr/bin/env python3
"""Visual test script for roof classifier.

Loads random images from each class in the data directory and shows classification results.
"""

import random
import sys
from pathlib import Path

from PIL import Image

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from roofline import RoofClassifier
from roofline.data import FOLDER_TO_CLASS


def get_sample_images(data_dir: Path, n_per_class: int = 3) -> list[tuple[Path, str]]:
    """Get random sample images from each class.

    Returns:
        List of (image_path, true_class) tuples
    """
    samples = []

    for folder_name, class_name in FOLDER_TO_CLASS.items():
        folder_path = data_dir / folder_name
        if not folder_path.exists():
            continue

        images = list(folder_path.glob("*.tif")) + list(folder_path.glob("*.tiff"))
        if not images:
            images = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg"))

        if images:
            selected = random.sample(images, min(n_per_class, len(images)))
            for img_path in selected:
                samples.append((img_path, class_name))

    random.shuffle(samples)
    return samples


def main():
    # Find data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Expected structure:")
        print("  data/Flat_data/")
        print("  data/Gable_hip_other/")
        print("  data/Complex_data/")
        sys.exit(1)

    # Load classifier
    print("Loading classifier...")
    classifier = RoofClassifier()
    print(f"Using device: {classifier.device}")
    print()

    # Get sample images
    samples = get_sample_images(data_dir, n_per_class=3)
    if not samples:
        print("No images found in data directory")
        sys.exit(1)

    print(f"Testing {len(samples)} images\n")
    print("-" * 70)

    # Track accuracy
    correct = 0
    total = 0
    class_stats = {c: {"correct": 0, "total": 0} for c in ["flat", "gable", "complex"]}

    for img_path, true_class in samples:
        # Load and classify
        img = Image.open(img_path)
        result = classifier.classify(img)

        pred_class = result["class"]
        confidence = result["confidence"]

        # Check if correct
        is_correct = pred_class == true_class
        if is_correct:
            correct += 1
            class_stats[true_class]["correct"] += 1
        total += 1
        class_stats[true_class]["total"] += 1

        # Display result
        status = "✓" if is_correct else "✗"
        print(f"{status} {img_path.name[:40]:<40}")
        print(f"  True: {true_class:<10} Pred: {pred_class:<10} Conf: {confidence:.2%}")

        # Show all probabilities
        probs_str = "  ".join(
            f"{c}: {p:.1%}" for c, p in sorted(result["probabilities"].items())
        )
        print(f"  [{probs_str}]")
        print()

    # Summary
    print("-" * 70)
    print(f"Overall Accuracy: {correct}/{total} ({correct / total:.1%})")
    print()
    print("Per-class accuracy:")
    for cls in ["flat", "gable", "complex"]:
        stats = class_stats[cls]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            print(f"  {cls:<10}: {stats['correct']}/{stats['total']} ({acc:.1%})")


if __name__ == "__main__":
    main()
