"""Unified trainer for full ASL recognition pipeline."""

from pathlib import Path
from typing import Optional

from loguru import logger

from src.training.train_classifier import ClassifierTrainer
from src.training.train_pose import PoseTrainer
from src.utils.constants import DATA_DIR, OUTPUTS_DIR


class ASLTrainer:
    """
    Unified trainer for the full ASL recognition pipeline.

    Orchestrates:
    1. YOLO26-pose fine-tuning for hand detection
    2. Keypoint extraction from ASL dataset
    3. Classifier training on extracted keypoints

    Example:
        >>> trainer = ASLTrainer()
        >>> trainer.train_full_pipeline()
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.

        Args:
            data_dir: Root data directory.
            output_dir: Root output directory.
        """
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else OUTPUTS_DIR

    def train_full_pipeline(
        self,
        # Pose training settings
        pose_model: str = "yolo26n-pose.pt",
        pose_epochs: int = 100,
        pose_batch_size: int = 16,
        # Classifier training settings
        classifier_type: str = "mlp",
        classifier_epochs: int = 50,
        classifier_batch_size: int = 64,
        # General settings
        device: Optional[str] = None,
        skip_pose: bool = False,
        skip_preprocessing: bool = False,
    ) -> dict:
        """
        Run full training pipeline.

        Args:
            pose_model: Base YOLO26-pose model.
            pose_epochs: Epochs for pose training.
            pose_batch_size: Batch size for pose training.
            classifier_type: Classifier architecture.
            classifier_epochs: Epochs for classifier training.
            classifier_batch_size: Batch size for classifier training.
            device: Device to train on.
            skip_pose: Skip pose fine-tuning (use pretrained).
            skip_preprocessing: Skip data preprocessing.

        Returns:
            Dictionary with training results.
        """
        results = {}

        # Stage 1: Fine-tune YOLO26-pose (optional)
        if not skip_pose:
            logger.info("=" * 60)
            logger.info("Stage 1: Fine-tuning YOLO26-pose for hand detection")
            logger.info("=" * 60)

            pose_trainer = PoseTrainer(
                model=pose_model,
                data="hand-keypoints.yaml",
                output_dir=self.output_dir / "pose_training",
            )

            pose_results = pose_trainer.train(
                epochs=pose_epochs,
                batch_size=pose_batch_size,
                device=device,
            )

            results["pose"] = {
                "mAP50": float(pose_results.results_dict.get("metrics/mAP50(P)", 0)),
                "mAP50-95": float(pose_results.results_dict.get("metrics/mAP50-95(P)", 0)),
            }
        else:
            logger.info("Skipping pose training (using pretrained model)")

        # Stage 2: Extract keypoints from SignAlphaSet
        if not skip_preprocessing:
            logger.info("=" * 60)
            logger.info("Stage 2: Extracting keypoints from SignAlphaSet")
            logger.info("=" * 60)

            from src.data.preprocess import DataPreprocessor

            preprocessor = DataPreprocessor(
                data_dir=self.data_dir,
                pose_model="weights/yolo26-pose-hands.pt" if not skip_pose else pose_model,
            )

            # Extract keypoints
            preprocessor.process_signalphaset()

            # Create splits
            preprocessor.create_classifier_dataset()

            results["preprocessing"] = {"status": "completed"}
        else:
            logger.info("Skipping preprocessing (using existing keypoints)")

        # Stage 3: Train classifier
        logger.info("=" * 60)
        logger.info("Stage 3: Training ASL classifier")
        logger.info("=" * 60)

        classifier_trainer = ClassifierTrainer(
            model_type=classifier_type,
            device=device,
        )

        classifier_dir = self.data_dir / "processed" / "classifier_dataset"
        history = classifier_trainer.train(
            data_dir=classifier_dir,
            epochs=classifier_epochs,
            batch_size=classifier_batch_size,
            output_dir=self.output_dir / "classifier_training",
        )

        # Evaluate on test set
        eval_results = classifier_trainer.evaluate(classifier_dir, split="test")

        results["classifier"] = {
            "best_val_accuracy": max(history["val_accuracy"]),
            "test_accuracy": eval_results["accuracy"],
            "epochs_trained": len(history["train_loss"]),
        }

        # Summary
        logger.info("=" * 60)
        logger.info("Training Pipeline Complete!")
        logger.info("=" * 60)
        logger.info(f"Results: {results}")

        return results

    def train_pose_only(
        self,
        model: str = "yolo26n-pose.pt",
        epochs: int = 100,
        **kwargs,
    ) -> dict:
        """Train only the pose model."""
        trainer = PoseTrainer(
            model=model,
            output_dir=self.output_dir / "pose_training",
        )
        results = trainer.train(epochs=epochs, **kwargs)
        return {"pose": results}

    def train_classifier_only(
        self,
        data_dir: Optional[Path] = None,
        model_type: str = "mlp",
        epochs: int = 50,
        **kwargs,
    ) -> dict:
        """Train only the classifier."""
        trainer = ClassifierTrainer(model_type=model_type, **kwargs)

        data_dir = data_dir or (self.data_dir / "processed" / "classifier_dataset")
        history = trainer.train(
            data_dir=data_dir,
            epochs=epochs,
            output_dir=self.output_dir / "classifier_training",
        )

        return {"classifier": history}


def main():
    """CLI entry point for unified trainer."""
    import argparse

    parser = argparse.ArgumentParser(description="Train ASL recognition pipeline")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    # Pipeline options
    parser.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")

    # Pose training
    parser.add_argument("--pose-model", default="yolo26n-pose.pt")
    parser.add_argument("--pose-epochs", type=int, default=100)

    # Classifier training
    parser.add_argument("--classifier-type", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--classifier-epochs", type=int, default=50)

    # General
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    trainer = ASLTrainer(data_dir=args.data_dir, output_dir=args.output_dir)

    trainer.train_full_pipeline(
        pose_model=args.pose_model,
        pose_epochs=args.pose_epochs,
        classifier_type=args.classifier_type,
        classifier_epochs=args.classifier_epochs,
        device=args.device,
        skip_pose=args.skip_pose,
        skip_preprocessing=args.skip_preprocessing,
    )


if __name__ == "__main__":
    main()
