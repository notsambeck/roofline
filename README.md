# Roofline

Simple PyTorch CNN for classifying roof types from satellite imagery.

**Classes:** flat, gable, complex, bug (4 classes)

## Installation

```bash
uv sync
```

## Usage

```python
from roofline import RoofClassifier
from PIL import Image

# Load trained model
classifier = RoofClassifier()

# Classify single image
img = Image.open("roof.tif")
result = classifier.classify(img)
# {"class": "gable", "confidence": 0.92, "probabilities": {...}}

# Classify batch
results = classifier.classify_batch([img1, img2, img3])
```

## Training

Train on your own dataset:

```bash
uv run python -m roofline.train --data /path/to/dataset --epochs 20 --output weights/model.pt
```

Expected dataset structure:
```
dataset/
├── flat/
├── gable/
├── complex/
└── bug/
```

Options:
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--val-split`: Validation split ratio (default: 0.2)
- `--device`: Device to train on (cuda/mps/cpu, auto-detected)

## Testing

Run unit tests:
```bash
uv run pytest tests/
```

Run visual test on sample images:
```bash
uv run python scripts/visual_test.py
```

## Model Architecture

Simple CNN with 4 conv blocks (~500K parameters):

```
Input (3, 224, 224)
  ↓ Conv2d(3→32) + BatchNorm + ReLU + MaxPool
  ↓ Conv2d(32→64) + BatchNorm + ReLU + MaxPool
  ↓ Conv2d(64→128) + BatchNorm + ReLU + MaxPool
  ↓ Conv2d(128→256) + BatchNorm + ReLU + MaxPool
  ↓ AdaptiveAvgPool2d(1)
  ↓ Linear(256→4)
Output (4 classes)
```
