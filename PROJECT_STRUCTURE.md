# YOLO26-ASL Project Architecture

## Directory Structure (Enterprise ML Standards)

```
yolo26-asl/
│
├── .github/                          # GitHub-specific configs
│   ├── workflows/
│   │   ├── ci.yml                    # Continuous Integration
│   │   ├── release.yml               # Auto-release on tag
│   │   └── docs.yml                  # Documentation build
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── dependabot.yml
│
├── configs/                          # Hydra/OmegaConf configuration
│   ├── __init__.py
│   ├── config.yaml                   # Main config entry point
│   ├── data/
│   │   ├── signalphaset.yaml         # SignAlphaSet dataset config
│   │   ├── hand_keypoints.yaml       # Ultralytics hand keypoints
│   │   └── combined.yaml             # Combined dataset config
│   ├── model/
│   │   ├── yolo26n_pose.yaml         # YOLO26-nano pose config
│   │   ├── yolo26s_pose.yaml         # YOLO26-small pose config
│   │   ├── classifier_mlp.yaml       # MLP classifier
│   │   └── classifier_transformer.yaml
│   ├── training/
│   │   ├── default.yaml
│   │   ├── fast_dev.yaml             # Quick debug training
│   │   └── production.yaml           # Full training config
│   └── inference/
│       ├── realtime.yaml
│       └── batch.yaml
│
├── src/                              # Main source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py               # Dataset downloaders
│   │   ├── preprocess.py             # Data preprocessing pipeline
│   │   ├── dataset.py                # PyTorch Dataset classes
│   │   ├── augmentation.py           # Albumentations augmentations
│   │   └── keypoint_utils.py         # Keypoint normalization/transforms
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hand_pose.py              # YOLO26-pose wrapper
│   │   ├── classifier.py             # ASL letter classifier
│   │   ├── pipeline.py               # End-to-end inference pipeline
│   │   └── components/
│   │       ├── __init__.py
│   │       ├── attention.py          # Attention modules
│   │       └── heads.py              # Classification heads
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Unified trainer class
│   │   ├── train_pose.py             # YOLO26-pose fine-tuning
│   │   ├── train_classifier.py       # Classifier training
│   │   ├── callbacks.py              # Training callbacks
│   │   ├── losses.py                 # Custom loss functions
│   │   └── optimizers.py             # Optimizer configs
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py              # YOLO26 vs YOLO11 benchmarks
│   │   ├── metrics.py                # Custom metrics (per-letter accuracy)
│   │   └── confusion_matrix.py       # Visualization
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py              # Single image/batch predictor
│   │   ├── realtime.py               # Webcam real-time inference
│   │   ├── video.py                  # Video file processing
│   │   └── export.py                 # Model export (ONNX, TensorRT)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                 # Structured logging (loguru)
│       ├── visualization.py          # Plotting utilities
│       ├── config.py                 # Config loading helpers
│       ├── device.py                 # Device selection (CPU/GPU/MPS)
│       └── constants.py              # Project constants
│
├── app/                              # Gradio web application
│   ├── __init__.py
│   ├── main.py                       # App entry point
│   ├── gradio_app.py                 # Gradio interface
│   ├── components/
│   │   ├── __init__.py
│   │   ├── webcam.py                 # Webcam component
│   │   ├── upload.py                 # Image upload component
│   │   └── results.py                # Results display
│   └── static/
│       ├── style.css
│       └── asl_reference.png
│
├── tests/                            # Pytest test suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   ├── unit/
│   │   ├── test_dataset.py
│   │   ├── test_models.py
│   │   └── test_utils.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_inference.py
│   └── e2e/
│       └── test_gradio_app.py
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_yolo26_pose_training.ipynb
│   ├── 03_classifier_training.ipynb
│   ├── 04_benchmark_yolo26_vs_yolo11.ipynb
│   └── 05_kaggle_submission.ipynb    # Kaggle-ready notebook
│
├── scripts/                          # Shell scripts for automation
│   ├── setup.sh                      # Environment setup
│   ├── download_data.sh              # Download all datasets
│   ├── train_all.sh                  # Full training pipeline
│   ├── benchmark.sh                  # Run benchmarks
│   └── export_models.sh              # Export to all formats
│
├── docker/                           # Docker configurations
│   ├── Dockerfile                    # CPU inference
│   ├── Dockerfile.gpu                # GPU training/inference
│   ├── Dockerfile.hf                 # HuggingFace Spaces
│   └── docker-compose.yml
│
├── docs/                             # Documentation
│   ├── index.md
│   ├── installation.md
│   ├── quickstart.md
│   ├── architecture.md
│   ├── training.md
│   ├── inference.md
│   ├── api/
│   │   └── reference.md
│   └── assets/
│
├── assets/                           # Static assets for README
│   ├── images/
│   │   ├── architecture.png
│   │   ├── demo.png
│   │   └── benchmark_chart.png
│   └── gifs/
│       └── demo.gif
│
├── weights/                          # Model weights (git-ignored)
│   └── .gitkeep
│
├── outputs/                          # Training outputs (git-ignored)
│   └── .gitkeep
│
├── data/                             # Data directory (git-ignored)
│   ├── raw/
│   ├── processed/
│   └── .gitkeep
│
│── .gitignore
├── .pre-commit-config.yaml           # Pre-commit hooks
├── .env.example                      # Environment variables template
├── pyproject.toml                    # Project metadata & tools config
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Development dependencies
├── Makefile                          # Common commands
├── README.md                         # Main documentation
├── CONTRIBUTING.md                   # Contribution guidelines
├── CHANGELOG.md                      # Version history
├── LICENSE                           # Apache 2.0
└── setup.py                          # Legacy setup (optional)
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         YOLO26-ASL Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────┐    ┌────────┐ │
│  │  Input   │───▶│  YOLO26-pose    │───▶│  Keypoint    │───▶│  ASL   │ │
│  │  Image   │    │  Hand Detection │    │  Classifier  │    │ Letter │ │
│  └──────────┘    │  (21 keypoints) │    │  (MLP/Trans) │    └────────┘ │
│                  └─────────────────┘    └──────────────┘                │
│                         │                      │                         │
│                         ▼                      ▼                         │
│                  ┌─────────────┐        ┌─────────────┐                 │
│                  │ RLE Loss    │        │ CrossEntropy│                 │
│                  │ (Pose)      │        │ + Label     │                 │
│                  └─────────────┘        │ Smoothing   │                 │
│                                         └─────────────┘                 │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Key YOLO26 Features:                                                    │
│  • NMS-free end-to-end architecture                                     │
│  • 43% faster CPU inference vs YOLO11                                   │
│  • RLE (Residual Log-Likelihood Estimation) for precise keypoints       │
│  • DFL-free design for edge deployment                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
SignAlphaSet (26K images)     Ultralytics Hand Keypoints (27K images)
         │                                    │
         ▼                                    ▼
    ┌─────────┐                        ┌─────────┐
    │ Letters │                        │ Generic │
    │ A-Z +   │                        │  Hand   │
    │ Gestures│                        │  Poses  │
    └────┬────┘                        └────┬────┘
         │                                  │
         └──────────┬───────────────────────┘
                    ▼
            ┌───────────────┐
            │   Combined    │
            │   Dataset     │
            │ (53K images)  │
            └───────┬───────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
    ┌─────────┐          ┌─────────┐
    │  Train  │          │   Val   │
    │  (80%)  │          │  (20%)  │
    └─────────┘          └─────────┘
```

## Tech Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Object Detection | YOLO26-pose | Latest Ultralytics, NMS-free, RLE |
| Keypoint Classifier | PyTorch MLP/Transformer | Lightweight, fast inference |
| Configuration | Hydra + OmegaConf | Industry standard, composable |
| Experiment Tracking | MLflow / W&B | Reproducibility |
| Web App | Gradio | Easy HuggingFace deployment |
| Testing | Pytest + Coverage | Quality assurance |
| CI/CD | GitHub Actions | Automated testing & release |
| Containerization | Docker | Reproducible environments |
| Documentation | MkDocs Material | Beautiful docs |
