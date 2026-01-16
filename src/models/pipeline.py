"""End-to-end ASL recognition pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from loguru import logger

from src.data.augmentation import KeypointNormalizer
from src.models.classifier import ASLClassifier
from src.models.hand_pose import HandPoseDetector
from src.utils.constants import WEIGHTS_DIR
from src.utils.device import get_device


@dataclass
class ASLPrediction:
    """Single ASL prediction result."""

    letter: str
    confidence: float
    class_idx: int
    keypoints: np.ndarray
    bbox: np.ndarray
    hand_confidence: float

    def __repr__(self) -> str:
        return f"ASLPrediction('{self.letter}', conf={self.confidence:.2%})"


class ASLPipeline:
    """
    End-to-end ASL recognition pipeline.

    Combines YOLO26-pose for hand detection and keypoint extraction
    with a classifier for letter recognition.

    Example:
        >>> pipeline = ASLPipeline()
        >>> predictions = pipeline.predict("image.jpg")
        >>> for pred in predictions:
        ...     print(f"{pred.letter}: {pred.confidence:.1%}")
    """

    def __init__(
        self,
        pose_model: Union[str, Path] = "yolo26n-pose.pt",
        classifier_model: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        pose_conf: float = 0.5,
        classifier_conf: float = 0.5,
    ):
        """
        Initialize pipeline.

        Args:
            pose_model: Path to YOLO26-pose model.
            classifier_model: Path to ASL classifier (None for default).
            device: Device to run on.
            pose_conf: Minimum confidence for pose detection.
            classifier_conf: Minimum confidence for classification.
        """
        self.device = device or str(get_device())
        self.pose_conf = pose_conf
        self.classifier_conf = classifier_conf

        # Initialize pose detector
        self.pose_detector = HandPoseDetector(
            model_path=pose_model,
            device=self.device,
            conf_threshold=pose_conf,
        )

        # Initialize classifier
        if classifier_model is None:
            classifier_model = WEIGHTS_DIR / "asl_classifier.pt"

        if Path(classifier_model).exists():
            self.classifier = ASLClassifier.load(classifier_model, device=self.device)
            logger.info(f"Loaded classifier from {classifier_model}")
        else:
            logger.warning(
                f"Classifier not found at {classifier_model}. "
                "Run training first or provide a valid model path."
            )
            self.classifier = None

        # Keypoint normalizer
        self.normalizer = KeypointNormalizer(method="center")

        logger.info(f"ASL Pipeline initialized on {self.device}")

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        max_hands: int = 2,
    ) -> list[ASLPrediction]:
        """
        Run full pipeline on an image.

        Args:
            source: Image path or numpy array (BGR).
            max_hands: Maximum number of hands to process.

        Returns:
            List of ASLPrediction objects.
        """
        if self.classifier is None:
            raise RuntimeError("Classifier not loaded. Train or load a model first.")

        # Detect hands
        detections = self.pose_detector.detect(source, max_hands=max_hands)

        if not detections:
            return []

        predictions = []
        for detection in detections:
            if not detection.is_valid():
                continue

            # Normalize keypoints
            keypoints = self.normalizer(detection.keypoints_normalized)

            # Convert to tensor
            kpts_tensor = torch.from_numpy(keypoints.flatten()).float()
            kpts_tensor = kpts_tensor.to(self.device)

            # Classify
            class_idx, conf, letter = self.classifier.predict(kpts_tensor)

            if conf >= self.classifier_conf:
                prediction = ASLPrediction(
                    letter=letter,
                    confidence=conf,
                    class_idx=class_idx,
                    keypoints=detection.keypoints,
                    bbox=detection.bbox,
                    hand_confidence=detection.confidence,
                )
                predictions.append(prediction)

        return predictions

    def predict_batch(
        self,
        sources: list[Union[str, np.ndarray]],
        max_hands: int = 2,
    ) -> list[list[ASLPrediction]]:
        """
        Run pipeline on multiple images.

        Args:
            sources: List of image paths or arrays.
            max_hands: Maximum hands per image.

        Returns:
            List of prediction lists.
        """
        all_predictions = []
        for source in sources:
            preds = self.predict(source, max_hands=max_hands)
            all_predictions.append(preds)
        return all_predictions

    def predict_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        display: bool = False,
        max_hands: int = 2,
    ) -> list[list[ASLPrediction]]:
        """
        Run pipeline on video file.

        Args:
            video_path: Path to video file.
            output_path: Optional path to save annotated video.
            display: Whether to display video during processing.
            max_hands: Maximum hands per frame.

        Returns:
            List of predictions per frame.
        """
        from src.utils.visualization import draw_hand_keypoints, draw_prediction

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        all_predictions = []
        frame_idx = 0

        logger.info(f"Processing video: {total_frames} frames at {fps} FPS")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run prediction
                predictions = self.predict(frame, max_hands=max_hands)
                all_predictions.append(predictions)

                # Draw annotations
                if writer or display:
                    annotated = frame.copy()

                    for pred in predictions:
                        # Draw keypoints
                        annotated = draw_hand_keypoints(annotated, pred.keypoints)

                        # Draw prediction
                        x1, y1 = int(pred.bbox[0]), int(pred.bbox[1])
                        annotated = draw_prediction(
                            annotated,
                            pred.letter,
                            pred.confidence,
                            position=(x1, max(y1 - 10, 30)),
                        )

                    if writer:
                        writer.write(annotated)

                    if display:
                        cv2.imshow("ASL Recognition", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                frame_idx += 1
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames")

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        if output_path:
            logger.info(f"Saved annotated video to {output_path}")

        return all_predictions

    def warmup(self) -> None:
        """Warmup models for optimal inference speed."""
        self.pose_detector.warmup()

        if self.classifier is not None:
            dummy = torch.zeros(1, 63).to(self.device)
            with torch.no_grad():
                self.classifier(dummy)

        logger.debug("Pipeline warmup complete")

    def benchmark(self, num_iterations: int = 100) -> dict[str, float]:
        """
        Benchmark pipeline speed.

        Args:
            num_iterations: Number of iterations for timing.

        Returns:
            Dictionary with timing results.
        """
        import time

        # Generate dummy input
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Warmup
        self.warmup()
        for _ in range(10):
            self.predict(dummy_image)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.predict(dummy_image)
            times.append(time.perf_counter() - start)

        times = np.array(times) * 1000  # Convert to ms

        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "fps": float(1000 / np.mean(times)),
        }


def main():
    """CLI entry point for prediction."""
    import argparse

    parser = argparse.ArgumentParser(description="ASL Recognition Pipeline")
    parser.add_argument("source", help="Image or video path")
    parser.add_argument("--pose-model", default="yolo26n-pose.pt")
    parser.add_argument("--classifier", default=None)
    parser.add_argument("--output", "-o", help="Output path for annotated video")
    parser.add_argument("--display", action="store_true", help="Display results")

    args = parser.parse_args()

    pipeline = ASLPipeline(
        pose_model=args.pose_model,
        classifier_model=args.classifier,
    )

    source = Path(args.source)

    if source.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        pipeline.predict_video(
            source,
            output_path=args.output,
            display=args.display,
        )
    else:
        predictions = pipeline.predict(str(source))
        for pred in predictions:
            print(f"Detected: {pred.letter} ({pred.confidence:.1%})")


if __name__ == "__main__":
    main()
