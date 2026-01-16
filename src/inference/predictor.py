"""ASL prediction interface."""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from src.models.pipeline import ASLPipeline, ASLPrediction


class ASLPredictor:
    """
    High-level ASL prediction interface.

    Provides simplified API for ASL recognition.

    Example:
        >>> predictor = ASLPredictor()
        >>> result = predictor("image.jpg")
        >>> print(result)  # "A" or None
    """

    def __init__(
        self,
        pose_model: str = "yolo26n-pose.pt",
        classifier_model: Optional[str] = None,
        device: Optional[str] = None,
        conf_threshold: float = 0.5,
    ):
        """
        Initialize predictor.

        Args:
            pose_model: Path to pose model.
            classifier_model: Path to classifier model.
            device: Device to run on.
            conf_threshold: Minimum confidence threshold.
        """
        self.pipeline = ASLPipeline(
            pose_model=pose_model,
            classifier_model=classifier_model,
            device=device,
            classifier_conf=conf_threshold,
        )
        self.conf_threshold = conf_threshold

    def __call__(
        self,
        image: Union[str, Path, np.ndarray],
        return_all: bool = False,
    ) -> Union[str, list[ASLPrediction], None]:
        """
        Predict ASL letter from image.

        Args:
            image: Image path or numpy array.
            return_all: If True, return all predictions.

        Returns:
            Predicted letter string, list of predictions, or None.
        """
        predictions = self.pipeline.predict(image)

        if not predictions:
            return [] if return_all else None

        if return_all:
            return predictions

        # Return highest confidence prediction
        best = max(predictions, key=lambda p: p.confidence)
        return best.letter

    def predict_with_confidence(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> tuple[Optional[str], float]:
        """
        Predict with confidence score.

        Returns:
            Tuple of (letter, confidence) or (None, 0.0).
        """
        predictions = self.pipeline.predict(image)

        if not predictions:
            return None, 0.0

        best = max(predictions, key=lambda p: p.confidence)
        return best.letter, best.confidence

    def predict_batch(
        self,
        images: list[Union[str, np.ndarray]],
    ) -> list[Optional[str]]:
        """
        Predict on multiple images.

        Args:
            images: List of image paths or arrays.

        Returns:
            List of predicted letters.
        """
        results = []
        for image in images:
            results.append(self(image))
        return results

    def warmup(self) -> None:
        """Warmup models."""
        self.pipeline.warmup()

    def benchmark(self, num_iterations: int = 100) -> dict:
        """Benchmark inference speed."""
        return self.pipeline.benchmark(num_iterations)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="ASL Predictor")
    parser.add_argument("source", help="Image path")
    parser.add_argument("--pose-model", default="yolo26n-pose.pt")
    parser.add_argument("--classifier", default=None)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    predictor = ASLPredictor(
        pose_model=args.pose_model,
        classifier_model=args.classifier,
        device=args.device,
    )

    letter, confidence = predictor.predict_with_confidence(args.source)
    if letter:
        print(f"Predicted: {letter} ({confidence:.1%})")
    else:
        print("No hand detected")


if __name__ == "__main__":
    main()
