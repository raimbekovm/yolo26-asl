"""Data augmentation for keypoint classification."""

from typing import Optional

import numpy as np


class KeypointAugmentor:
    """
    Augmentation pipeline for hand keypoints.

    Applies transformations that preserve keypoint relationships
    while increasing data diversity.

    Example:
        >>> augmentor = KeypointAugmentor(rotation=15, scale=(0.8, 1.2))
        >>> augmented = augmentor(keypoints)
    """

    def __init__(
        self,
        rotation: float = 15.0,
        scale: tuple[float, float] = (0.9, 1.1),
        translate: float = 0.1,
        noise_std: float = 0.01,
        keypoint_dropout: float = 0.0,
        horizontal_flip: float = 0.0,
        p: float = 0.5,
    ):
        """
        Initialize augmentor.

        Args:
            rotation: Max rotation angle in degrees.
            scale: Scale range (min, max).
            translate: Max translation as fraction of image size.
            noise_std: Standard deviation of Gaussian noise.
            keypoint_dropout: Probability of dropping each keypoint.
            horizontal_flip: Probability of horizontal flip.
            p: Probability of applying any augmentation.
        """
        self.rotation = rotation
        self.scale = scale
        self.translate = translate
        self.noise_std = noise_std
        self.keypoint_dropout = keypoint_dropout
        self.horizontal_flip = horizontal_flip
        self.p = p

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to keypoints.

        Args:
            keypoints: Array of shape (21, 3) with x, y, visibility.

        Returns:
            Augmented keypoints of same shape.
        """
        if np.random.random() > self.p:
            return keypoints

        keypoints = keypoints.copy()

        # Extract coordinates and visibility
        coords = keypoints[:, :2]  # (21, 2)
        visibility = keypoints[:, 2] if keypoints.shape[1] > 2 else None

        # Apply transforms
        if self.rotation > 0:
            coords = self._rotate(coords, np.random.uniform(-self.rotation, self.rotation))

        if self.scale[0] != 1.0 or self.scale[1] != 1.0:
            coords = self._scale(coords, np.random.uniform(*self.scale))

        if self.translate > 0:
            coords = self._translate(coords, self.translate)

        if self.noise_std > 0:
            coords = self._add_noise(coords, self.noise_std)

        if self.horizontal_flip > 0 and np.random.random() < self.horizontal_flip:
            coords = self._flip_horizontal(coords)

        # Apply keypoint dropout
        if self.keypoint_dropout > 0 and visibility is not None:
            dropout_mask = np.random.random(len(visibility)) < self.keypoint_dropout
            visibility = visibility.copy()
            visibility[dropout_mask] = 0

        # Reconstruct keypoints
        if visibility is not None:
            keypoints = np.column_stack([coords, visibility])
        else:
            keypoints = coords

        return keypoints

    def _rotate(self, coords: np.ndarray, angle: float) -> np.ndarray:
        """Rotate keypoints around center."""
        # Center coordinates
        center = coords.mean(axis=0)
        centered = coords - center

        # Rotation matrix
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])

        # Apply rotation
        rotated = centered @ R.T

        return rotated + center

    def _scale(self, coords: np.ndarray, scale: float) -> np.ndarray:
        """Scale keypoints from center."""
        center = coords.mean(axis=0)
        centered = coords - center
        scaled = centered * scale
        return scaled + center

    def _translate(self, coords: np.ndarray, max_translate: float) -> np.ndarray:
        """Translate keypoints."""
        tx = np.random.uniform(-max_translate, max_translate)
        ty = np.random.uniform(-max_translate, max_translate)
        return coords + np.array([tx, ty])

    def _add_noise(self, coords: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise to coordinates."""
        noise = np.random.normal(0, std, coords.shape)
        return coords + noise

    def _flip_horizontal(self, coords: np.ndarray) -> np.ndarray:
        """Flip keypoints horizontally."""
        coords = coords.copy()
        coords[:, 0] = 1.0 - coords[:, 0]  # Assuming normalized coordinates
        return coords


class KeypointNormalizer:
    """
    Normalize keypoints for consistent model input.

    Supports different normalization strategies:
    - 'minmax': Scale to [0, 1] range
    - 'center': Center around origin with unit scale
    - 'wrist': Normalize relative to wrist position

    Example:
        >>> normalizer = KeypointNormalizer(method='center')
        >>> normalized = normalizer(keypoints)
    """

    def __init__(self, method: str = "center"):
        """
        Initialize normalizer.

        Args:
            method: Normalization method ('minmax', 'center', 'wrist').
        """
        self.method = method

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints.

        Args:
            keypoints: Array of shape (21, 3) or (21, 2).

        Returns:
            Normalized keypoints.
        """
        keypoints = keypoints.copy()
        coords = keypoints[:, :2]
        has_visibility = keypoints.shape[1] > 2

        if self.method == "minmax":
            coords = self._minmax_normalize(coords)
        elif self.method == "center":
            coords = self._center_normalize(coords)
        elif self.method == "wrist":
            coords = self._wrist_normalize(coords)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if has_visibility:
            return np.column_stack([coords, keypoints[:, 2]])
        return coords

    def _minmax_normalize(self, coords: np.ndarray) -> np.ndarray:
        """Scale to [0, 1] range."""
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        scale = max_vals - min_vals
        scale[scale == 0] = 1  # Avoid division by zero
        return (coords - min_vals) / scale

    def _center_normalize(self, coords: np.ndarray) -> np.ndarray:
        """Center and scale to unit variance."""
        center = coords.mean(axis=0)
        centered = coords - center
        scale = np.abs(centered).max()
        if scale > 0:
            centered /= scale
        return centered

    def _wrist_normalize(self, coords: np.ndarray) -> np.ndarray:
        """Normalize relative to wrist (keypoint 0)."""
        wrist = coords[0]
        relative = coords - wrist

        # Scale by hand size (distance to middle fingertip)
        middle_tip = coords[12]
        scale = np.linalg.norm(middle_tip - wrist)
        if scale > 0:
            relative /= scale

        return relative


def get_train_augmentation(config: Optional[dict] = None) -> KeypointAugmentor:
    """Get default training augmentation."""
    defaults = {
        "rotation": 15.0,
        "scale": (0.9, 1.1),
        "translate": 0.1,
        "noise_std": 0.01,
        "keypoint_dropout": 0.05,
        "horizontal_flip": 0.0,  # Don't flip for ASL (left/right matters)
        "p": 0.5,
    }

    if config:
        defaults.update(config)

    return KeypointAugmentor(**defaults)


def get_val_augmentation() -> KeypointNormalizer:
    """Get validation transform (normalization only)."""
    return KeypointNormalizer(method="center")
