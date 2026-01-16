.PHONY: help install install-dev setup lint format test test-cov clean train predict app docker benchmark docs

# ==================== Variables ====================
PYTHON := python
PIP := pip
PROJECT := yolo26-asl
SRC := src
TESTS := tests

# Colors for terminal output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m  # No Color

# ==================== Help ====================
help:  ## Show this help message
	@echo "$(BLUE)YOLO26-ASL$(NC) - American Sign Language Recognition"
	@echo ""
	@echo "$(GREEN)Usage:$(NC)"
	@echo "  make $(YELLOW)<target>$(NC)"
	@echo ""
	@echo "$(GREEN)Targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# ==================== Installation ====================
install:  ## Install production dependencies
	$(PIP) install -e .

install-dev:  ## Install development dependencies
	$(PIP) install -e ".[all]"
	pre-commit install

install-app:  ## Install with Gradio app dependencies
	$(PIP) install -e ".[app]"

setup: install-dev  ## Full development setup
	@echo "$(GREEN)Setting up development environment...$(NC)"
	mkdir -p data/{raw,processed} weights outputs
	@echo "$(GREEN)Setup complete!$(NC)"

# ==================== Code Quality ====================
lint:  ## Run linters (ruff + mypy)
	@echo "$(BLUE)Running ruff...$(NC)"
	ruff check $(SRC) $(TESTS)
	@echo "$(BLUE)Running mypy...$(NC)"
	mypy $(SRC)

format:  ## Format code (black + ruff)
	@echo "$(BLUE)Formatting with black...$(NC)"
	black $(SRC) $(TESTS) app
	@echo "$(BLUE)Sorting imports with ruff...$(NC)"
	ruff check --fix --select I $(SRC) $(TESTS) app

check: lint  ## Run all checks (lint + type check)
	@echo "$(GREEN)All checks passed!$(NC)"

# ==================== Testing ====================
test:  ## Run tests
	pytest $(TESTS) -v

test-cov:  ## Run tests with coverage
	pytest $(TESTS) -v --cov=$(SRC) --cov-report=html --cov-report=term-missing

test-fast:  ## Run fast tests only (exclude slow)
	pytest $(TESTS) -v -m "not slow"

test-gpu:  ## Run GPU tests only
	pytest $(TESTS) -v -m "gpu"

# ==================== Data ====================
download-data:  ## Download all datasets
	@echo "$(BLUE)Downloading datasets...$(NC)"
	$(PYTHON) -m src.data.download --all

download-signalphaset:  ## Download SignAlphaSet dataset
	$(PYTHON) -m src.data.download --dataset signalphaset

download-hand-keypoints:  ## Download Ultralytics hand keypoints dataset
	$(PYTHON) -m src.data.download --dataset hand-keypoints

preprocess:  ## Preprocess datasets
	$(PYTHON) -m src.data.preprocess

# ==================== Training ====================
train:  ## Train full pipeline (pose + classifier)
	@echo "$(BLUE)Starting training pipeline...$(NC)"
	$(PYTHON) -m src.training.trainer

train-pose:  ## Train YOLO26-pose model only
	$(PYTHON) -m src.training.train_pose

train-classifier:  ## Train ASL classifier only
	$(PYTHON) -m src.training.train_classifier

train-fast:  ## Quick training run for debugging
	$(PYTHON) -m src.training.trainer training=fast_dev

# ==================== Inference ====================
predict:  ## Run prediction on sample image
	$(PYTHON) -m src.inference.predictor --source assets/images/sample.jpg

predict-video:  ## Run prediction on video file
	$(PYTHON) -m src.inference.video --source data/test_video.mp4

webcam:  ## Run real-time webcam inference
	$(PYTHON) -m src.inference.realtime

# ==================== Evaluation ====================
benchmark:  ## Run YOLO26 vs YOLO11 benchmark
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(PYTHON) -m src.evaluation.benchmark

evaluate:  ## Evaluate model on test set
	$(PYTHON) -m src.evaluation.metrics

# ==================== Application ====================
app:  ## Launch Gradio web application
	@echo "$(GREEN)Launching Gradio app...$(NC)"
	$(PYTHON) -m app.main

app-share:  ## Launch Gradio with public link
	$(PYTHON) -m app.main --share

# ==================== Export ====================
export-onnx:  ## Export model to ONNX format
	$(PYTHON) -m src.inference.export --format onnx

export-tensorrt:  ## Export model to TensorRT format
	$(PYTHON) -m src.inference.export --format engine

export-all:  ## Export to all formats
	$(PYTHON) -m src.inference.export --format all

# ==================== Docker ====================
docker-build:  ## Build Docker image (CPU)
	docker build -t $(PROJECT):latest -f docker/Dockerfile .

docker-build-gpu:  ## Build Docker image (GPU)
	docker build -t $(PROJECT):gpu -f docker/Dockerfile.gpu .

docker-run:  ## Run Docker container
	docker run -it --rm -p 7860:7860 $(PROJECT):latest

docker-run-gpu:  ## Run Docker container with GPU
	docker run -it --rm --gpus all -p 7860:7860 $(PROJECT):gpu

docker-compose:  ## Run with docker-compose
	docker-compose -f docker/docker-compose.yml up

# ==================== Documentation ====================
docs:  ## Build documentation
	mkdocs build

docs-serve:  ## Serve documentation locally
	mkdocs serve

# ==================== Cleanup ====================
clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean  ## Clean everything including data and weights
	rm -rf data/processed/
	rm -rf outputs/
	rm -rf runs/
	rm -rf wandb/
	rm -rf mlruns/

# ==================== CI/CD ====================
ci: lint test  ## Run CI pipeline locally
	@echo "$(GREEN)CI pipeline passed!$(NC)"

release:  ## Create release (bump version, tag, push)
	@echo "$(RED)Please use: git tag -a vX.Y.Z -m 'Release vX.Y.Z' && git push --tags$(NC)"
