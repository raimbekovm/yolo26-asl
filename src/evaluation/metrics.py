"""Evaluation metrics for ASL recognition."""

from typing import Optional

import numpy as np
from loguru import logger


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional class names.

    Returns:
        Dictionary with metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    from src.utils.constants import ASL_CLASSES

    if class_names is None:
        class_names = ASL_CLASSES

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Per-class metrics
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class"] = report

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Top confused pairs
    metrics["top_confusions"] = _get_top_confusions(cm, class_names)

    return metrics


def _get_top_confusions(
    cm: np.ndarray,
    class_names: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Get top confused class pairs."""
    confusions = []

    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true": class_names[i],
                    "predicted": class_names[j],
                    "count": int(cm[i, j]),
                })

    confusions.sort(key=lambda x: x["count"], reverse=True)
    return confusions[:top_k]


def compute_pose_metrics(
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """
    Compute pose estimation metrics.

    Args:
        pred_keypoints: Predicted keypoints (N, 21, 2).
        gt_keypoints: Ground truth keypoints (N, 21, 2).
        threshold: Distance threshold for correct keypoint.

    Returns:
        Dictionary with pose metrics.
    """
    n_samples = len(pred_keypoints)
    n_keypoints = pred_keypoints.shape[1]

    # Per-keypoint accuracy
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=-1)
    correct = distances < threshold

    pck = correct.mean()  # Percentage of Correct Keypoints
    per_keypoint_pck = correct.mean(axis=0)

    # Mean Per Joint Position Error
    mpjpe = distances.mean()

    metrics = {
        "pck": float(pck),
        "mpjpe": float(mpjpe),
        "per_keypoint_pck": per_keypoint_pck.tolist(),
    }

    return metrics


def print_metrics_summary(metrics: dict) -> None:
    """Print formatted metrics summary."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")
    print(f"Macro F1 Score:   {metrics['f1_macro']:.2%}")
    print(f"Weighted F1:      {metrics['f1_weighted']:.2%}")

    print("\nTop Confusions:")
    for conf in metrics.get("top_confusions", []):
        print(f"  {conf['true']} -> {conf['predicted']}: {conf['count']} times")

    print("=" * 50)
