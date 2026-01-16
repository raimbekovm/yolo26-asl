"""Device selection and configuration utilities."""

import os
from typing import Literal, Optional

import torch
from loguru import logger


DeviceType = Literal["auto", "cpu", "cuda", "mps"]


def get_device(
    device: DeviceType = "auto",
    device_id: Optional[int] = None,
    verbose: bool = True,
) -> torch.device:
    """
    Get the best available device for PyTorch.

    Args:
        device: Device type preference.
            - "auto": Automatically select best available
            - "cpu": Force CPU
            - "cuda": Use NVIDIA GPU
            - "mps": Use Apple Silicon GPU
        device_id: Specific GPU ID (for multi-GPU systems).
        verbose: Whether to log device information.

    Returns:
        torch.device: Selected device.

    Example:
        >>> device = get_device("auto")
        >>> model = model.to(device)
    """
    if device == "auto":
        if torch.cuda.is_available():
            selected = "cuda"
        elif torch.backends.mps.is_available():
            selected = "mps"
        else:
            selected = "cpu"
    else:
        selected = device

    # Validate selection
    if selected == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            selected = "cpu"
        elif device_id is not None:
            if device_id >= torch.cuda.device_count():
                logger.warning(
                    f"GPU {device_id} not found, using GPU 0. "
                    f"Available: {torch.cuda.device_count()}"
                )
                device_id = 0
            selected = f"cuda:{device_id}"

    elif selected == "mps":
        if not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            selected = "cpu"

    torch_device = torch.device(selected)

    if verbose:
        _log_device_info(torch_device)

    return torch_device


def _log_device_info(device: torch.device) -> None:
    """Log detailed device information."""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        logger.info(f"Using CUDA device: {gpu_name} ({gpu_memory:.1f} GB)")

    elif device.type == "mps":
        logger.info("Using Apple Silicon MPS device")

    else:
        logger.info("Using CPU device")


def get_optimal_workers(device: torch.device) -> int:
    """Get optimal number of DataLoader workers for the device."""
    cpu_count = os.cpu_count() or 4

    if device.type == "cuda":
        # For GPU training, use more workers for data loading
        return min(8, cpu_count)
    if device.type == "mps":
        # MPS works well with fewer workers
        return min(4, cpu_count)
    # CPU training - leave some cores for model
    return max(1, cpu_count - 2)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, use deterministic algorithms (slower).
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 2.0+
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)

    logger.debug(f"Random seed set to {seed} (deterministic={deterministic})")


def get_device_info() -> dict:
    """Get comprehensive device information as a dictionary."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_count": os.cpu_count(),
    }

    if torch.cuda.is_available():
        info.update(
            {
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ],
            }
        )

    return info
