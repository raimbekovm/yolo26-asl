"""Data preprocessing pipeline for ASL recognition."""

import json
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.utils.constants import ASL_CLASSES, CLASS_TO_IDX, DATA_DIR


class DataPreprocessor:
    """
    Preprocess datasets for ASL recognition training.

    Pipeline:
    1. Extract keypoints from SignAlphaSet using YOLO26-pose
    2. Create unified dataset format
    3. Split into train/val/test

    Example:
        >>> preprocessor = DataPreprocessor()
        >>> preprocessor.process_signalphaset()
        >>> preprocessor.create_classifier_dataset()
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        pose_model: str = "yolo26n-pose.pt",
    ):
        """
        Initialize preprocessor.

        Args:
            data_dir: Root data directory.
            pose_model: YOLO pose model for keypoint extraction.
        """
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.pose_model_name = pose_model
        self._pose_model = None

    @property
    def pose_model(self):
        """Lazy load pose model."""
        if self._pose_model is None:
            from ultralytics import YOLO
            logger.info(f"Loading pose model: {self.pose_model_name}")
            self._pose_model = YOLO(self.pose_model_name)
        return self._pose_model

    def process_signalphaset(
        self,
        output_dir: Optional[Path] = None,
        conf_threshold: float = 0.5,
    ) -> Path:
        """
        Extract keypoints from SignAlphaSet images.

        Args:
            output_dir: Output directory for processed data.
            conf_threshold: Minimum confidence for keypoint detection.

        Returns:
            Path to processed dataset.
        """
        input_dir = self.raw_dir / "signalphaset" / "images"
        output_dir = output_dir or (self.processed_dir / "signalphaset_keypoints")
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_dir.exists():
            raise FileNotFoundError(
                f"SignAlphaSet not found at {input_dir}. "
                "Run 'make download-signalphaset' first."
            )

        logger.info(f"Processing SignAlphaSet from {input_dir}")

        all_samples = []

        # Process each class directory
        for class_dir in sorted(input_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            if class_name not in CLASS_TO_IDX:
                logger.warning(f"Unknown class: {class_name}, skipping")
                continue

            class_idx = CLASS_TO_IDX[class_name]
            logger.info(f"Processing class: {class_name} ({class_idx})")

            # Get all images
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

            for img_path in tqdm(images, desc=class_name):
                sample = self._extract_keypoints(
                    img_path, class_name, class_idx, conf_threshold
                )
                if sample is not None:
                    all_samples.append(sample)

        # Save as DataFrame
        df = pd.DataFrame(all_samples)
        csv_path = output_dir / "keypoints.csv"
        df.to_csv(csv_path, index=False)

        # Save keypoints as numpy array
        keypoints = np.array([s["keypoints"] for s in all_samples])
        labels = np.array([s["class_idx"] for s in all_samples])
        np.save(output_dir / "keypoints.npy", keypoints)
        np.save(output_dir / "labels.npy", labels)

        logger.info(
            f"Processed {len(all_samples)} samples, saved to {output_dir}"
        )

        return output_dir

    def _extract_keypoints(
        self,
        img_path: Path,
        class_name: str,
        class_idx: int,
        conf_threshold: float,
    ) -> Optional[dict]:
        """Extract keypoints from a single image."""
        try:
            # Run pose detection
            results = self.pose_model(str(img_path), verbose=False)

            if len(results) == 0 or results[0].keypoints is None:
                return None

            keypoints = results[0].keypoints

            # Get first detection (assuming one hand per image)
            if keypoints.xy.shape[0] == 0:
                return None

            kpts = keypoints.xy[0].cpu().numpy()  # (21, 2)

            # Get confidence if available
            if keypoints.conf is not None:
                conf = keypoints.conf[0].cpu().numpy()  # (21,)
            else:
                conf = np.ones(21)

            # Filter by confidence
            if np.mean(conf) < conf_threshold:
                return None

            # Combine into (21, 3) array
            kpts_with_conf = np.column_stack([kpts, conf])

            # Normalize keypoints to [0, 1]
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            kpts_normalized = kpts_with_conf.copy()
            kpts_normalized[:, 0] /= w
            kpts_normalized[:, 1] /= h

            return {
                "image_path": str(img_path),
                "class_name": class_name,
                "class_idx": class_idx,
                "keypoints": kpts_normalized.flatten().tolist(),
                "confidence": float(np.mean(conf)),
                "width": w,
                "height": h,
            }

        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")
            return None

    def create_classifier_dataset(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        random_state: int = 42,
    ) -> dict[str, Path]:
        """
        Create train/val/test splits for classifier training.

        Args:
            input_dir: Directory with keypoints.npy and labels.npy.
            output_dir: Output directory for splits.
            train_ratio: Training set ratio.
            val_ratio: Validation set ratio.
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary with paths to train/val/test data.
        """
        from sklearn.model_selection import train_test_split

        input_dir = input_dir or (self.processed_dir / "signalphaset_keypoints")
        output_dir = output_dir or (self.processed_dir / "classifier_dataset")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        keypoints = np.load(input_dir / "keypoints.npy")
        labels = np.load(input_dir / "labels.npy")

        logger.info(f"Loaded {len(labels)} samples")

        # Split data
        test_ratio = 1 - train_ratio - val_ratio

        X_train, X_temp, y_train, y_temp = train_test_split(
            keypoints, labels,
            test_size=(1 - train_ratio),
            random_state=random_state,
            stratify=labels,
        )

        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size),
            random_state=random_state,
            stratify=y_temp,
        )

        # Save splits
        paths = {}
        for split, (X, y) in [
            ("train", (X_train, y_train)),
            ("val", (X_val, y_val)),
            ("test", (X_test, y_test)),
        ]:
            split_dir = output_dir / split
            split_dir.mkdir(exist_ok=True)

            np.save(split_dir / "keypoints.npy", X)
            np.save(split_dir / "labels.npy", y)
            paths[split] = split_dir

            logger.info(f"{split}: {len(y)} samples")

        # Save metadata
        metadata = {
            "num_classes": len(ASL_CLASSES),
            "classes": ASL_CLASSES,
            "keypoint_dims": 63,  # 21 * 3
            "splits": {
                "train": len(y_train),
                "val": len(y_val),
                "test": len(y_test),
            },
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return paths

    def create_yolo_pose_dataset(
        self,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Create YOLO-format pose dataset for fine-tuning.

        Combines hand keypoints data with custom annotations.

        Args:
            output_dir: Output directory.

        Returns:
            Path to dataset YAML file.
        """
        output_dir = output_dir or (self.processed_dir / "yolo_pose_dataset")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset YAML
        yaml_content = f"""# YOLO26 Hand Pose Dataset
# Auto-generated for ASL recognition

path: {output_dir}
train: images/train
val: images/val

# Keypoints
kpt_shape: [21, 3]
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Classes
names:
  0: hand
"""

        yaml_path = output_dir / "dataset.yaml"
        yaml_path.write_text(yaml_content)

        logger.info(f"Created YOLO pose dataset config at {yaml_path}")
        return yaml_path


def main():
    """CLI entry point for preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess ASL datasets")
    parser.add_argument(
        "--task",
        choices=["extract-keypoints", "create-splits", "all"],
        default="all",
    )
    parser.add_argument("--data-dir", type=Path, help="Data directory")
    parser.add_argument("--pose-model", default="yolo26n-pose.pt")

    args = parser.parse_args()

    preprocessor = DataPreprocessor(
        data_dir=args.data_dir,
        pose_model=args.pose_model,
    )

    if args.task in ["extract-keypoints", "all"]:
        preprocessor.process_signalphaset()

    if args.task in ["create-splits", "all"]:
        preprocessor.create_classifier_dataset()


if __name__ == "__main__":
    main()
