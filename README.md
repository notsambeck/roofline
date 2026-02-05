# Roofline

A simple PyTorch CNN for classifying roof types from satellite/aerial imagery.

## Classes

| Class | Description |
|-------|-------------|
| `flat` | Flat roofs |
| `gable` | Gable, hip, and other pitched roofs |
| `complex` | Complex roof structures |
| `bug` | Imagery artifacts or unclear cases |

## Example Results

![Test Results](test_results.png)

## Installation

```bash
# Clone the repository
git clone https://github.com/notsambeck/roofline.git
cd roofline

# Install with uv
uv sync

# Or with pip
pip install -e .
```

## Usage

```python
from roofline import RoofClassifier
from PIL import Image

# Load classifier (uses bundled weights if available)
classifier = RoofClassifier()

# Classify a single image
img = Image.open("roof.tif")
result = classifier.classify(img)
# {"class": "gable", "confidence": 0.92, "probabilities": {...}}

# Classify a batch of images
results = classifier.classify_batch([img1, img2, img3])
```

### Custom model weights

```python
classifier = RoofClassifier(model_path="path/to/your/model.pt")
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

### Training options

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | required | Path to dataset directory |
| `--output` | `weights/model.pt` | Output model path |
| `--epochs` | 20 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--val-split` | 0.2 | Validation split ratio |
| `--device` | auto | Device (cuda/mps/cpu) |

## Model Architecture

RoofNet is a simple CNN with 4 convolutional blocks (~390K parameters):

```
Input (3, 224, 224)
  │
  ├─ Conv2d(3→32, 3×3) + BatchNorm + ReLU + MaxPool2d
  ├─ Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool2d
  ├─ Conv2d(64→128, 3×3) + BatchNorm + ReLU + MaxPool2d
  ├─ Conv2d(128→256, 3×3) + BatchNorm + ReLU + MaxPool2d
  │
  ├─ AdaptiveAvgPool2d(1)
  └─ Linear(256→4)

Output: 4 class logits
```

## Development

```bash
# Run tests
uv run pytest tests/

# Run visual test on sample images
uv run python scripts/visual_test.py
```

## Dataset

This project was developed using the "Imagery dataset for rooftop detection and classification" containing 3,617 GeoTIFF images of rooftops from aerial imagery of Sofia, Bulgaria (10 cm/pixel resolution, captured 2020).

> Hristov, E., Petrova-Antonova, D., Petrov, A., Borukova, M., & Shirinyan, E. (2023). *Imagery dataset for rooftop detection and classification* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7633594

## License

MIT License

## Citation

If you use this code in your research, please cite both this repository and the dataset:

```bibtex
@software{roofline,
  title = {Roofline: Roof Type Classifier},
  url = {https://github.com/notsambeck/roofline}
}

@dataset{hristov_2023_7633594,
  author = {Hristov, Emil and Petrova-Antonova, Dessislava and Petrov, Alexander and Borukova, Milena and Shirinyan, Evgeny},
  title = {Imagery dataset for rooftop detection and classification},
  year = {2023},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.7633594},
  url = {https://doi.org/10.5281/zenodo.7633594}
}
```
