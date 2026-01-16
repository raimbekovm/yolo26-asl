"""PyTorch Dataset classes for ASL keypoint classification."""

from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.constants import ASL_CLASSES, DATA_DIR, NUM_KEYPOINTS


class ASLKeypointDataset(Dataset):
    """
    PyTorch Dataset for ASL keypoint classification.

    Loads pre-extracted keypoints and labels for classifier training.

    Example:
        >>> dataset = ASLKeypointDataset("data/processed/classifier_dataset/train")
        >>> keypoints, label = dataset[0]
        >>> print(keypoints.shape)  # (63,) or (21, 3)
        torch.Size([63])
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        flatten: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing keypoints.npy and labels.npy.
            transform: Optional transform to apply to keypoints.
            flatten: Whether to flatten keypoints to 1D (63,) or keep (21, 3).
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.flatten = flatten

        # Load data
        self.keypoints = np.load(self.data_dir / "keypoints.npy")
        self.labels = np.load(self.data_dir / "labels.npy")

        # Reshape if needed
        if self.keypoints.ndim == 2 and self.keypoints.shape[1] == 63:
            # Already flattened (N, 63) -> reshape to (N, 21, 3)
            self.keypoints = self.keypoints.reshape(-1, NUM_KEYPOINTS, 3)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        keypoints = self.keypoints[idx].copy()  # (21, 3)
        label = int(self.labels[idx])

        # Apply transform
        if self.transform is not None:
            keypoints = self.transform(keypoints)

        # Convert to tensor
        keypoints = torch.from_numpy(keypoints).float()

        # Flatten if requested
        if self.flatten:
            keypoints = keypoints.flatten()

        return keypoints, label

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        class_counts = np.bincount(self.labels, minlength=len(ASL_CLASSES))
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(ASL_CLASSES)
        return torch.from_numpy(class_weights).float()

    @property
    def num_classes(self) -> int:
        return len(ASL_CLASSES)

    @property
    def input_dim(self) -> int:
        return NUM_KEYPOINTS * 3 if self.flatten else (NUM_KEYPOINTS, 3)


class KeypointDataModule:
    """
    Data module for managing train/val/test dataloaders.

    Example:
        >>> dm = KeypointDataModule("data/processed/classifier_dataset")
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        transform_train: Optional[Callable] = None,
        transform_val: Optional[Callable] = None,
    ):
        """
        Initialize data module.

        Args:
            data_dir: Root directory with train/val/test subdirs.
            batch_size: Batch size for dataloaders.
            num_workers: Number of data loading workers.
            pin_memory: Whether to pin memory for GPU training.
            transform_train: Transform for training data.
            transform_val: Transform for validation/test data.
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform_train = transform_train
        self.transform_val = transform_val

        self.train_dataset: Optional[ASLKeypointDataset] = None
        self.val_dataset: Optional[ASLKeypointDataset] = None
        self.test_dataset: Optional[ASLKeypointDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage."""
        if stage in (None, "fit"):
            self.train_dataset = ASLKeypointDataset(
                self.data_dir / "train",
                transform=self.transform_train,
            )
            self.val_dataset = ASLKeypointDataset(
                self.data_dir / "val",
                transform=self.transform_val,
            )

        if stage in (None, "test"):
            self.test_dataset = ASLKeypointDataset(
                self.data_dir / "test",
                transform=self.transform_val,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @property
    def num_classes(self) -> int:
        return len(ASL_CLASSES)

    @property
    def input_dim(self) -> int:
        return NUM_KEYPOINTS * 3


class RealtimeKeypointBuffer:
    """
    Buffer for temporal smoothing in real-time inference.

    Stores recent keypoint predictions for stability filtering.

    Example:
        >>> buffer = RealtimeKeypointBuffer(window_size=5)
        >>> buffer.add(keypoints, prediction, confidence)
        >>> smoothed = buffer.get_smoothed_prediction()
    """

    def __init__(self, window_size: int = 5):
        """
        Initialize buffer.

        Args:
            window_size: Number of recent frames to store.
        """
        self.window_size = window_size
        self.keypoints: list[np.ndarray] = []
        self.predictions: list[int] = []
        self.confidences: list[float] = []

    def add(
        self,
        keypoints: np.ndarray,
        prediction: int,
        confidence: float,
    ) -> None:
        """Add a new frame to the buffer."""
        self.keypoints.append(keypoints)
        self.predictions.append(prediction)
        self.confidences.append(confidence)

        # Trim to window size
        if len(self.predictions) > self.window_size:
            self.keypoints.pop(0)
            self.predictions.pop(0)
            self.confidences.pop(0)

    def get_smoothed_prediction(self, method: str = "mode") -> tuple[int, float]:
        """
        Get smoothed prediction from buffer.

        Args:
            method: Smoothing method ('mode', 'weighted', 'last').

        Returns:
            Tuple of (prediction, confidence).
        """
        if not self.predictions:
            return -1, 0.0

        if method == "mode":
            # Most common prediction
            from collections import Counter
            counter = Counter(self.predictions)
            prediction = counter.most_common(1)[0][0]
            confidence = np.mean([
                c for p, c in zip(self.predictions, self.confidences)
                if p == prediction
            ])
        elif method == "weighted":
            # Weighted by confidence
            weighted_votes = {}
            for pred, conf in zip(self.predictions, self.confidences):
                weighted_votes[pred] = weighted_votes.get(pred, 0) + conf
            prediction = max(weighted_votes, key=weighted_votes.get)
            confidence = weighted_votes[prediction] / len(self.predictions)
        else:  # last
            prediction = self.predictions[-1]
            confidence = self.confidences[-1]

        return prediction, confidence

    def clear(self) -> None:
        """Clear the buffer."""
        self.keypoints.clear()
        self.predictions.clear()
        self.confidences.clear()

    def is_stable(self, min_consecutive: int = 3) -> bool:
        """Check if recent predictions are stable."""
        if len(self.predictions) < min_consecutive:
            return False
        return len(set(self.predictions[-min_consecutive:])) == 1
