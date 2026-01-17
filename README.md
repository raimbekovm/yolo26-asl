<p align="center">
  <h1 align="center">YOLO26-ASL</h1>
  <p align="center">
    <strong>Real-time American Sign Language Detection with YOLO26</strong>
  </p>
</p>

<p align="center">
  <a href="https://github.com/raimbekovm/yolo26-asl/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red.svg" alt="PyTorch">
  </a>
  <a href="https://docs.ultralytics.com/models/yolo26/">
    <img src="https://img.shields.io/badge/YOLO26-Ultralytics-purple.svg" alt="YOLO26">
  </a>
  <a href="https://huggingface.co/spaces/raimbekovm/yolo26-asl">
    <img src="https://img.shields.io/badge/Demo-HuggingFace-yellow.svg" alt="Demo">
  </a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/raimbekovm/yolo26-asl">
    <img src="https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-yellow?style=for-the-badge" alt="Live Demo">
  </a>
</p>

---

## Overview

**YOLO26-ASL** benchmarks [YOLO26](https://docs.ultralytics.com/models/yolo26/) against YOLO11 on American Sign Language (ASL) letter detection. This project trains and evaluates both models on the same dataset to provide an honest comparison.

### Key Features

| Feature | Description |
|---------|-------------|
| **YOLO26 vs YOLO11** | Fair benchmark comparison on ASL detection |
| **26 Classes** | A-Z letter detection with bounding boxes |
| **Real Benchmarks** | Actual training results on Kaggle T4 GPU |
| **Production Ready** | Gradio demo, Docker support, HuggingFace Spaces |

---

## Benchmark Results

### YOLO26n vs YOLO11n on ASL Detection

Trained on [American Sign Language Letters](https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters) dataset (504 train / 144 val / 72 test images).

| Metric | YOLO26n | YOLO11n | Winner |
|--------|---------|---------|--------|
| **mAP50** | 0.751 | **0.906** | YOLO11 |
| **mAP50-95** | 0.715 | **0.860** | YOLO11 |
| **GPU Inference** | 13.1 ms | **11.6 ms** | YOLO11 |
| **GPU FPS** | 76 | **86** | YOLO11 |
| **CPU Inference** | **122.8 ms** | 127.2 ms | YOLO26 |
| **CPU FPS** | **8.1** | 7.9 | YOLO26 |
| **Parameters** | **2.57M** | 2.62M | YOLO26 |

### Key Findings

- **YOLO11n achieves +15.5% higher mAP50** on this small dataset
- **YOLO26n is 3.6% faster on CPU** - beneficial for edge deployment
- Both models achieve real-time performance on GPU (>75 FPS)
- Small dataset (504 images) may favor YOLO11's architecture
- YOLO26 has slightly fewer parameters (2% less)

### Hardware & Training

- **GPU**: Tesla T4 (Kaggle)
- **Epochs**: 100
- **Batch Size**: 16
- **Image Size**: 640×640
- **Augmentation**: Mosaic + MixUp

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/raimbekovm/yolo26-asl.git
cd yolo26-asl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[app]"
```

### Run Demo

```bash
# Launch Gradio app
python -m app.main --share
```

### Download Trained Weights

```bash
# Option 1: Already included in weights/
ls weights/yolo26n_asl.pt

# Option 2: Download from HuggingFace
wget https://huggingface.co/spaces/raimbekovm/yolo26-asl/resolve/main/yolo26n_asl.pt
```

### Quick Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("weights/yolo26n_asl.pt")

# Run inference
results = model("path/to/hand_sign.jpg")
results[0].show()
```

---

## Training

### Run on Kaggle (Recommended)

1. Open [notebooks/kaggle_yolo26_asl.ipynb](notebooks/kaggle_yolo26_asl.ipynb)
2. Upload to Kaggle
3. Enable GPU accelerator (T4)
4. Run all cells

### Local Training

```bash
# Download dataset
pip install roboflow
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_KEY')
project = rf.workspace('david-lee-d0rhs').project('american-sign-language-letters')
dataset = project.version(6).download('yolov8')
"

# Train YOLO26
yolo detect train model=yolo26n.pt data=data/asl_dataset.yaml epochs=100 imgsz=640

# Train YOLO11 (baseline)
yolo detect train model=yolo11n.pt data=data/asl_dataset.yaml epochs=100 imgsz=640
```

---

## Project Structure

```
yolo26-asl/
├── app/                  # Gradio web application
├── configs/              # Configuration files
├── data/                 # Dataset configs
├── docker/               # Docker configurations
├── notebooks/            # Kaggle benchmark notebook
│   └── kaggle_yolo26_asl.ipynb
├── src/                  # Source code
├── tests/                # Unit tests
└── weights/              # Trained model weights
    └── yolo26n_asl.pt    # Trained on ASL dataset (5.2MB)
```

---

## Dataset

**American Sign Language Letters** from Roboflow Universe:
- **Classes**: 26 (A-Z)
- **Train**: 504 images
- **Validation**: 144 images
- **Test**: 72 images
- **Format**: YOLOv8 (bounding boxes)

---

## Docker

```bash
# Build image
docker build -t yolo26-asl -f docker/Dockerfile .

# Run container
docker run -p 7860:7860 yolo26-asl
```

---

## Citation

```bibtex
@software{yolo26_asl,
  author = {Murat Raimbekov},
  title = {YOLO26-ASL: ASL Detection Benchmark with YOLO26},
  year = {2026},
  url = {https://github.com/raimbekovm/yolo26-asl}
}
```

---

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO26 and YOLO11
- [Roboflow](https://roboflow.com/) for ASL dataset hosting
- [Kaggle](https://kaggle.com/) for free GPU compute

---

## License

Apache 2.0 License - see [LICENSE](LICENSE) file.

---

<p align="center">
  <strong>Built with Ultralytics YOLO26</strong><br>
  <a href="https://docs.ultralytics.com/models/yolo26/">Documentation</a> •
  <a href="https://github.com/raimbekovm/yolo26-asl/issues">Issues</a> •
  <a href="https://huggingface.co/spaces/raimbekovm/yolo26-asl">Demo</a>
</p>
