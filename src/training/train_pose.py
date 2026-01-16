"""YOLO26-pose fine-tuning for hand keypoint detection."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger


if TYPE_CHECKING:
    from ultralytics.engine.results import Results

from src.utils.constants import OUTPUTS_DIR, WEIGHTS_DIR


class PoseTrainer:
    """
    Fine-tune YOLO26-pose for hand keypoint detection.

    Uses Ultralytics training API with optimized settings for hands.

    Example:
        >>> trainer = PoseTrainer()
        >>> results = trainer.train(epochs=100)
    """

    def __init__(
        self,
        model: str = "yolo26n-pose.pt",
        data: str = "hand-keypoints.yaml",
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Base YOLO26-pose model to fine-tune.
            data: Dataset YAML configuration.
            output_dir: Directory for training outputs.
        """
        from ultralytics import YOLO

        self.model_name = model
        self.data = data
        self.output_dir = output_dir or OUTPUTS_DIR / "pose_training"

        logger.info(f"Loading base model: {model}")
        self.model = YOLO(model)

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        device: Optional[str] = None,
        workers: int = 8,
        project: Optional[str] = None,
        name: str = "yolo26_pose_hands",
        # Optimizer settings
        optimizer: str = "AdamW",
        lr0: float = 0.001,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        # Augmentation settings
        augment: bool = True,
        mosaic: float = 1.0,
        mixup: float = 0.0,
        # Loss weights
        pose: float = 12.0,
        box: float = 7.5,
        cls: float = 0.5,
        # Training settings
        patience: int = 50,
        save_period: int = 10,
        val: bool = True,
        plots: bool = True,
        resume: bool = False,
        **kwargs,
    ) -> "Results":
        """
        Run training.

        Args:
            epochs: Number of training epochs.
            batch_size: Batch size.
            imgsz: Input image size.
            device: Device to train on.
            workers: Number of data loading workers.
            project: Project name for outputs.
            name: Run name.
            optimizer: Optimizer type.
            lr0: Initial learning rate.
            lrf: Final learning rate factor.
            momentum: Optimizer momentum.
            weight_decay: Weight decay.
            augment: Enable augmentation.
            mosaic: Mosaic augmentation probability.
            mixup: MixUp augmentation probability.
            pose: Pose loss weight.
            box: Box loss weight.
            cls: Classification loss weight.
            patience: Early stopping patience.
            save_period: Save checkpoint every N epochs.
            val: Run validation during training.
            plots: Generate training plots.
            resume: Resume from last checkpoint.
            **kwargs: Additional arguments for YOLO train.

        Returns:
            Training results object.
        """
        project = project or str(self.output_dir)

        logger.info(f"Starting YOLO26-pose training for {epochs} epochs")
        logger.info(f"Dataset: {self.data}")
        logger.info(f"Output: {project}/{name}")

        results = self.model.train(
            data=self.data,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            workers=workers,
            project=project,
            name=name,
            # Optimizer
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            # Augmentation
            augment=augment,
            mosaic=mosaic,
            mixup=mixup,
            # Loss weights
            pose=pose,
            box=box,
            cls=cls,
            # Training
            patience=patience,
            save_period=save_period,
            val=val,
            plots=plots,
            resume=resume,
            exist_ok=True,
            **kwargs,
        )

        # Copy best weights
        best_weights = Path(project) / name / "weights" / "best.pt"
        if best_weights.exists():
            target = WEIGHTS_DIR / "yolo26-pose-hands.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy(best_weights, target)
            logger.info(f"Best weights saved to {target}")

        return results

    def validate(
        self,
        weights: Optional[str] = None,
        data: Optional[str] = None,
        split: str = "val",
        **kwargs,
    ) -> dict:
        """
        Run validation.

        Args:
            weights: Model weights path (None for current model).
            data: Dataset YAML (None for training data).
            split: Data split to validate on.
            **kwargs: Additional arguments.

        Returns:
            Validation metrics dictionary.
        """
        if weights:
            from ultralytics import YOLO

            model = YOLO(weights)
        else:
            model = self.model

        data = data or self.data

        results = model.val(data=data, split=split, **kwargs)

        metrics = {
            "mAP50": results.pose.map50,
            "mAP50-95": results.pose.map,
            "precision": results.pose.mp,
            "recall": results.pose.mr,
        }

        logger.info(
            f"Validation results: mAP50={metrics['mAP50']:.3f}, mAP50-95={metrics['mAP50-95']:.3f}"
        )

        return metrics

    def export(
        self,
        weights: Optional[str] = None,
        format: str = "onnx",
        imgsz: int = 640,
        half: bool = True,
        simplify: bool = True,
        **kwargs,
    ) -> Path:
        """
        Export model to deployment format.

        Args:
            weights: Model weights path.
            format: Export format (onnx, engine, coreml, tflite).
            imgsz: Input image size.
            half: Use FP16 precision.
            simplify: Simplify ONNX model.
            **kwargs: Additional arguments.

        Returns:
            Path to exported model.
        """
        if weights:
            from ultralytics import YOLO

            model = YOLO(weights)
        else:
            model = self.model

        logger.info(f"Exporting model to {format} format")

        export_path = model.export(
            format=format,
            imgsz=imgsz,
            half=half,
            simplify=simplify,
            **kwargs,
        )

        logger.info(f"Model exported to {export_path}")
        return Path(export_path)


def main():
    """CLI entry point for pose training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO26-pose for hand detection")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="Base model")
    parser.add_argument("--data", default="hand-keypoints.yaml", help="Dataset config")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="yolo26_pose_hands")

    args = parser.parse_args()

    trainer = PoseTrainer(model=args.model, data=args.data)
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
    )


if __name__ == "__main__":
    main()
