"""YOLO26-pose wrapper for hand keypoint detection."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from loguru import logger

from src.utils.constants import DEFAULT_CONF_THRESHOLD, DEFAULT_IMAGE_SIZE, NUM_KEYPOINTS


class HandPoseDetector:
    """
    YOLO26-pose wrapper for hand keypoint detection.

    Provides a clean interface for detecting hands and extracting
    21 keypoints per hand.

    Example:
        >>> detector = HandPoseDetector("yolo26n-pose.pt")
        >>> results = detector.detect("image.jpg")
        >>> for hand in results:
        ...     print(hand.keypoints.shape)  # (21, 3)
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "yolo26n-pose.pt",
        device: Optional[str] = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = 0.7,
        imgsz: int = DEFAULT_IMAGE_SIZE,
    ):
        """
        Initialize detector.

        Args:
            model_path: Path to YOLO pose model weights.
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto).
            conf_threshold: Minimum confidence for detections.
            iou_threshold: IoU threshold for NMS.
            imgsz: Input image size.
        """
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        # Load model
        logger.info(f"Loading YOLO26-pose model: {model_path}")
        self.model = YOLO(str(model_path))

        # Set device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Hand pose detector ready on {self.device}")

    def detect(
        self,
        source: Union[str, Path, np.ndarray],
        max_hands: int = 2,
        return_image: bool = False,
    ) -> list["HandDetection"]:
        """
        Detect hands and extract keypoints.

        Args:
            source: Image path or numpy array (BGR).
            max_hands: Maximum number of hands to return.
            return_image: Whether to include annotated image in results.

        Returns:
            List of HandDetection objects.
        """
        # Run inference
        results = self.model(
            source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        if not results or len(results) == 0:
            return []

        result = results[0]
        detections = []

        # Check if keypoints exist
        if result.keypoints is None or result.keypoints.xy.shape[0] == 0:
            return []

        # Process each detection
        num_detections = min(result.keypoints.xy.shape[0], max_hands)

        for i in range(num_detections):
            # Get keypoints (21, 2)
            kpts_xy = result.keypoints.xy[i].cpu().numpy()

            # Get confidence per keypoint if available
            if result.keypoints.conf is not None:
                kpts_conf = result.keypoints.conf[i].cpu().numpy()
            else:
                kpts_conf = np.ones(NUM_KEYPOINTS)

            # Combine into (21, 3)
            keypoints = np.column_stack([kpts_xy, kpts_conf])

            # Get bounding box if available
            if result.boxes is not None and len(result.boxes) > i:
                bbox = result.boxes.xyxy[i].cpu().numpy()
                box_conf = float(result.boxes.conf[i].cpu())
            else:
                # Compute bbox from keypoints
                valid_kpts = kpts_xy[kpts_conf > 0.5]
                if len(valid_kpts) > 0:
                    x_min, y_min = valid_kpts.min(axis=0)
                    x_max, y_max = valid_kpts.max(axis=0)
                    bbox = np.array([x_min, y_min, x_max, y_max])
                else:
                    bbox = np.zeros(4)
                box_conf = float(np.mean(kpts_conf))

            detection = HandDetection(
                keypoints=keypoints,
                bbox=bbox,
                confidence=box_conf,
                image_shape=result.orig_shape,
            )
            detections.append(detection)

        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)

        return detections[:max_hands]

    def detect_batch(
        self,
        sources: list[Union[str, np.ndarray]],
        max_hands: int = 2,
    ) -> list[list["HandDetection"]]:
        """
        Batch detection for multiple images.

        Args:
            sources: List of image paths or arrays.
            max_hands: Maximum hands per image.

        Returns:
            List of detection lists (one per image).
        """
        results = self.model(
            sources,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            stream=True,
        )

        all_detections = []
        for result in results:
            detections = self._process_result(result, max_hands)
            all_detections.append(detections)

        return all_detections

    def _process_result(self, result, max_hands: int) -> list["HandDetection"]:
        """Process a single result into HandDetection objects."""
        detections = []

        if result.keypoints is None or result.keypoints.xy.shape[0] == 0:
            return []

        num_detections = min(result.keypoints.xy.shape[0], max_hands)

        for i in range(num_detections):
            kpts_xy = result.keypoints.xy[i].cpu().numpy()
            kpts_conf = (
                result.keypoints.conf[i].cpu().numpy()
                if result.keypoints.conf is not None
                else np.ones(NUM_KEYPOINTS)
            )

            keypoints = np.column_stack([kpts_xy, kpts_conf])

            if result.boxes is not None and len(result.boxes) > i:
                bbox = result.boxes.xyxy[i].cpu().numpy()
                box_conf = float(result.boxes.conf[i].cpu())
            else:
                bbox = np.zeros(4)
                box_conf = float(np.mean(kpts_conf))

            detections.append(
                HandDetection(
                    keypoints=keypoints,
                    bbox=bbox,
                    confidence=box_conf,
                    image_shape=result.orig_shape,
                )
            )

        return sorted(detections, key=lambda x: x.confidence, reverse=True)[:max_hands]

    def warmup(self, imgsz: Optional[int] = None) -> None:
        """Warmup model with dummy input."""
        imgsz = imgsz or self.imgsz
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.detect(dummy)
        logger.debug("Model warmup complete")


class HandDetection:
    """
    Single hand detection result.

    Attributes:
        keypoints: Array of shape (21, 3) with x, y, confidence.
        bbox: Bounding box [x1, y1, x2, y2].
        confidence: Detection confidence.
        image_shape: Original image shape (height, width).
    """

    def __init__(
        self,
        keypoints: np.ndarray,
        bbox: np.ndarray,
        confidence: float,
        image_shape: tuple[int, int],
    ):
        self.keypoints = keypoints
        self.bbox = bbox
        self.confidence = confidence
        self.image_shape = image_shape

    @property
    def keypoints_xy(self) -> np.ndarray:
        """Get x, y coordinates only."""
        return self.keypoints[:, :2]

    @property
    def keypoints_conf(self) -> np.ndarray:
        """Get confidence scores only."""
        return self.keypoints[:, 2]

    @property
    def keypoints_normalized(self) -> np.ndarray:
        """Get keypoints normalized to [0, 1]."""
        h, w = self.image_shape
        normalized = self.keypoints.copy()
        normalized[:, 0] /= w
        normalized[:, 1] /= h
        return normalized

    @property
    def keypoints_flat(self) -> np.ndarray:
        """Get flattened keypoints for classifier input."""
        return self.keypoints_normalized.flatten()

    def is_valid(self, min_visible: int = 15, min_conf: float = 0.3) -> bool:
        """Check if detection has enough visible keypoints."""
        visible = (self.keypoints_conf > min_conf).sum()
        return visible >= min_visible

    def __repr__(self) -> str:
        return (
            f"HandDetection(confidence={self.confidence:.2f}, "
            f"visible_kpts={(self.keypoints_conf > 0.5).sum()}/21)"
        )
