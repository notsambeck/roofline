"""Training script for RoofNet classifier."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .data import RoofDataset, get_train_transforms, get_val_transforms
from .model import RoofNet


def train_epoch(
    model: RoofNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def validate(
    model: RoofNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train(
    data_dir: str | Path,
    output_path: str | Path,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    val_split: float = 0.2,
    device: str | None = None,
):
    """Train the RoofNet classifier."""
    # Setup device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load dataset
    full_dataset = RoofDataset(data_dir, transform=None, train=True)
    print(f"Loaded {len(full_dataset)} images")

    # Print class distribution
    class_counts = [0] * 4
    for _, label in full_dataset.samples:
        class_counts[label] += 1
    for i, count in enumerate(class_counts):
        print(f"  {RoofDataset.CLASSES[i]}: {count}")

    # Split into train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply different transforms to train and val
    class TransformDataset:
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img_path, class_idx = self.dataset.dataset.samples[self.dataset.indices[idx]]
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, class_idx

    train_wrapped = TransformDataset(train_dataset, get_train_transforms())
    val_wrapped = TransformDataset(val_dataset, get_val_transforms())

    train_loader = DataLoader(train_wrapped, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_wrapped, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train size: {train_size}, Val size: {val_size}")

    # Create model (ResNet18 with frozen backbone)
    model = RoofNet(freeze_backbone=True).to(device)
    print(f"Using ResNet18 backbone (frozen)")
    print(f"Trainable parameters: {model.count_parameters():,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # Training loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"  Saved best model (val_acc: {val_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train RoofNet classifier")
    parser.add_argument("--data", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output", type=str, default="weights/model.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train(
        data_dir=args.data,
        output_path=output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device=args.device,
    )


if __name__ == "__main__":
    main()
