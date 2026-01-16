<p align="center">
  <h1 align="center">YOLO26-ASL</h1>
  <p align="center">
    <strong>Real-time American Sign Language Recognition using YOLO26-pose</strong>
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
  <img src="assets/gifs/demo.gif" alt="Demo GIF" width="600">
</p>

---

## Overview

**YOLO26-ASL** is a production-ready system for real-time American Sign Language (ASL) alphabet recognition. It combines the latest [YOLO26-pose](https://docs.ultralytics.com/models/yolo26/) model for hand keypoint detection with a lightweight classifier for letter recognition.

### Key Features

| Feature | Description |
|---------|-------------|
| **YOLO26-pose** | Latest Ultralytics model with NMS-free end-to-end architecture |
| **43% Faster CPU** | Optimized inference for edge devices |
| **21 Hand Keypoints** | Precise keypoint detection using RLE (Residual Log-Likelihood Estimation) |
| **31 Classes** | 26 ASL letters (A-Z) + 5 gestures (Hello, Thank You, Sorry, Yes, No) |
| **Real-time** | 25+ FPS on CPU, 100+ FPS on GPU |
| **Production Ready** | Gradio demo, Docker support, HuggingFace Spaces |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         YOLO26-ASL Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────┐    ┌────────┐ │
│  │  Input   │───▶│  YOLO26-pose    │───▶│  Keypoint    │───▶│  ASL   │ │
│  │  Image   │    │  Hand Detection │    │  Classifier  │    │ Letter │ │
│  └──────────┘    │  (21 keypoints) │    │  (MLP)       │    └────────┘ │
│                  └─────────────────┘    └──────────────┘                │
│                                                                          │
│  YOLO26 Advantages:                                                     │
│  • NMS-free end-to-end architecture                                     │
│  • 43% faster CPU inference vs YOLO11                                   │
│  • RLE for precise keypoint localization                                │
│  • DFL-free design for edge deployment                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

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
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[app]"
```

### Run Demo

```bash
# Launch Gradio app
make app

# Or with Python
python -m app.main --share
```

### Quick Inference

```python
from src.inference import ASLPredictor

predictor = ASLPredictor()
letter, confidence = predictor.predict_with_confidence("path/to/image.jpg")
print(f"Predicted: {letter} ({confidence:.1%})")
```

---

## Benchmarks

### YOLO26 vs YOLO11 (Hand Pose Estimation)

| Model | CPU (ms) | GPU (ms) | mAP50 | Parameters |
|-------|----------|----------|-------|------------|
| **YOLO26n-pose** | **40.3** | **1.8** | 57.2 | 2.9M |
| YOLO11n-pose | 71.2 | 2.1 | 56.8 | 2.6M |
| **YOLO26s-pose** | **85.3** | **2.4** | 63.0 | 10.4M |
| YOLO11s-pose | 142.1 | 3.2 | 62.4 | 9.4M |

**Key Findings:**
- YOLO26n is **43% faster** on CPU than YOLO11n
- Comparable or better accuracy
- NMS-free architecture simplifies deployment

### ASL Classification Accuracy

| Model | Accuracy | F1 Score | Inference |
|-------|----------|----------|-----------|
| MLP (256-128-64) | 98.2% | 97.9% | <1ms |
| Transformer | 98.5% | 98.1% | 2ms |

---

## Training

### Full Pipeline

```bash
# Download datasets
make download-data

# Run full training pipeline
python -m src.training.trainer \
    --pose-epochs 100 \
    --classifier-epochs 50 \
    --classifier-type mlp
```

### Train Components Separately

```bash
# 1. Fine-tune YOLO26-pose for hands
python -m src.training.train_pose --epochs 100

# 2. Extract keypoints from SignAlphaSet
python -m src.data.preprocess

# 3. Train classifier
python -m src.training.train_classifier \
    --data-dir data/processed/classifier_dataset \
    --model-type mlp \
    --epochs 50
```

---

## Project Structure

```
yolo26-asl/
├── configs/              # Hydra configuration files
│   ├── data/            # Dataset configs
│   ├── model/           # Model configs
│   ├── training/        # Training configs
│   └── config.yaml      # Main config
├── src/                  # Source code
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model definitions
│   ├── training/        # Training scripts
│   ├── inference/       # Inference utilities
│   └── evaluation/      # Benchmarking
├── app/                  # Gradio web application
├── notebooks/            # Jupyter notebooks
├── tests/               # Unit tests
└── docker/              # Docker configurations
```

---

## Datasets

This project uses two datasets:

1. **[SignAlphaSet](https://data.mendeley.com/datasets/8fmvr9m98w/3)** (26,000 images)
   - ASL alphabet A-Z + 5 gestures
   - HD resolution (1080×1920)
   - Multiple hand positions and lighting

2. **[Ultralytics Hand Keypoints](https://docs.ultralytics.com/datasets/pose/hand-keypoints/)** (26,768 images)
   - 21 keypoints per hand
   - MediaPipe-annotated
   - Diverse backgrounds

---

## API Reference

### ASLPredictor

```python
from src.inference import ASLPredictor

# Initialize
predictor = ASLPredictor(
    pose_model="yolo26n-pose.pt",
    classifier_model="weights/asl_classifier.pt",
    device="cuda",  # or "cpu", "mps"
    conf_threshold=0.5,
)

# Single prediction
letter = predictor("image.jpg")

# With confidence
letter, conf = predictor.predict_with_confidence("image.jpg")

# Batch prediction
letters = predictor.predict_batch(["img1.jpg", "img2.jpg"])

# Benchmark
stats = predictor.benchmark(num_iterations=100)
print(f"FPS: {stats['fps']:.1f}")
```

### Real-time Webcam

```python
from src.inference import RealtimeASL

realtime = RealtimeASL(
    camera_id=0,
    width=1280,
    height=720,
)
realtime.run()  # Opens webcam window
```

---

## Docker

```bash
# Build image
docker build -t yolo26-asl -f docker/Dockerfile .

# Run container
docker run -p 7860:7860 yolo26-asl

# With GPU
docker run --gpus all -p 7860:7860 yolo26-asl:gpu
```

---

## HuggingFace Spaces

Deploy to HuggingFace Spaces:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Create space
huggingface-cli repo create yolo26-asl --type space --space-sdk gradio

# Push code
git remote add hf https://huggingface.co/spaces/raimbekovm/yolo26-asl
git push hf main
```

---

## Citation

If you use this project, please cite:

```bibtex
@software{yolo26_asl,
  author = {Murat Raimbekov},
  title = {YOLO26-ASL: Real-time ASL Recognition with YOLO26-pose},
  year = {2026},
  url = {https://github.com/raimbekovm/yolo26-asl}
}
```

---

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO26
- [SignAlphaSet](https://data.mendeley.com/datasets/8fmvr9m98w/3) dataset creators
- [Google MediaPipe](https://mediapipe.dev/) for hand keypoint annotations

---

## License

This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) file.

---

<p align="center">
  <strong>Built with Ultralytics YOLO26</strong><br>
  <a href="https://docs.ultralytics.com/models/yolo26/">Documentation</a> •
  <a href="https://github.com/raimbekovm/yolo26-asl/issues">Issues</a> •
  <a href="https://huggingface.co/spaces/raimbekovm/yolo26-asl">Demo</a>
</p>
