"""Visualization utilities for keypoints, predictions, and metrics."""

from pathlib import Path
from typing import Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger

from src.utils.constants import (
    ASL_CLASSES,
    COLORS,
    FINGER_COLORS,
    HAND_KEYPOINTS,
    HAND_SKELETON,
)


def draw_hand_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    conf_threshold: float = 0.5,
    radius: int = 5,
    thickness: int = 2,
    draw_skeleton: bool = True,
    colored_fingers: bool = True,
) -> np.ndarray:
    """
    Draw hand keypoints and skeleton on an image.

    Args:
        image: Input image (BGR format).
        keypoints: Keypoints array of shape (21, 2) or (21, 3).
        confidence: Optional confidence scores of shape (21,).
        conf_threshold: Minimum confidence to draw a keypoint.
        radius: Keypoint circle radius.
        thickness: Line thickness for skeleton.
        draw_skeleton: Whether to draw skeleton connections.
        colored_fingers: Whether to use different colors per finger.

    Returns:
        Image with drawn keypoints and skeleton.
    """
    image = image.copy()
    kpts = keypoints[:, :2].astype(int)

    # Get confidence from keypoints if not provided separately
    if confidence is None and keypoints.shape[1] == 3:
        confidence = keypoints[:, 2]
    elif confidence is None:
        confidence = np.ones(len(kpts))

    # Draw skeleton first (so keypoints are on top)
    if draw_skeleton:
        for i, j in HAND_SKELETON:
            if confidence[i] > conf_threshold and confidence[j] > conf_threshold:
                pt1 = tuple(kpts[i])
                pt2 = tuple(kpts[j])

                # Get color based on finger
                if colored_fingers:
                    color = _get_finger_color(i, j)
                else:
                    color = COLORS["cyan"]

                cv2.line(image, pt1, pt2, color, thickness)

    # Draw keypoints
    for idx, (pt, conf) in enumerate(zip(kpts, confidence)):
        if conf > conf_threshold:
            if colored_fingers:
                color = _get_keypoint_color(idx)
            else:
                color = COLORS["green"]

            cv2.circle(image, tuple(pt), radius, color, -1)
            cv2.circle(image, tuple(pt), radius, COLORS["black"], 1)

    return image


def _get_finger_color(i: int, j: int) -> tuple[int, int, int]:
    """Get color for a skeleton connection based on finger."""
    finger_ranges = {
        "thumb": range(0, 5),
        "index": range(5, 9),
        "middle": range(9, 13),
        "ring": range(13, 17),
        "pinky": range(17, 21),
    }

    for finger, rng in finger_ranges.items():
        if i in rng or j in rng:
            return FINGER_COLORS.get(finger, COLORS["white"])

    return COLORS["white"]


def _get_keypoint_color(idx: int) -> tuple[int, int, int]:
    """Get color for a keypoint based on its finger."""
    if idx == 0:
        return FINGER_COLORS["wrist"]
    elif idx <= 4:
        return FINGER_COLORS["thumb"]
    elif idx <= 8:
        return FINGER_COLORS["index"]
    elif idx <= 12:
        return FINGER_COLORS["middle"]
    elif idx <= 16:
        return FINGER_COLORS["ring"]
    else:
        return FINGER_COLORS["pinky"]


def draw_prediction(
    image: np.ndarray,
    prediction: str,
    confidence: float,
    position: tuple[int, int] = (10, 50),
    font_scale: float = 1.5,
    thickness: int = 2,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Draw prediction text on image with background.

    Args:
        image: Input image.
        prediction: Predicted class name.
        confidence: Prediction confidence (0-1).
        position: Text position (x, y).
        font_scale: Font scale.
        thickness: Text thickness.
        bg_color: Background color.
        text_color: Text color.

    Returns:
        Image with prediction text.
    """
    image = image.copy()
    text = f"{prediction}: {confidence:.1%}"

    # Get text size for background
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Draw background rectangle
    x, y = position
    padding = 10
    cv2.rectangle(
        image,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        bg_color,
        -1,
    )

    # Draw text
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)

    return image


def draw_fps(
    image: np.ndarray,
    fps: float,
    position: tuple[int, int] = (10, 30),
) -> np.ndarray:
    """Draw FPS counter on image."""
    image = image.copy()
    text = f"FPS: {fps:.1f}"

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLORS["green"],
        2,
    )

    return image


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[list[str]] = None,
    normalize: bool = True,
    figsize: tuple[int, int] = (14, 12),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot confusion matrix for ASL classification.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        classes: Class names (defaults to ASL_CLASSES).
        normalize: Whether to normalize values.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    from sklearn.metrics import confusion_matrix

    if classes is None:
        classes = ASL_CLASSES

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("ASL Classification Confusion Matrix", fontsize=14)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


def plot_training_curves(
    history: dict[str, list[float]],
    figsize: tuple[int, int] = (12, 4),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', etc.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    n_plots = len([k for k in history.keys() if "train" in k])
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    plot_idx = 0
    for key in history:
        if "train" in key:
            metric = key.replace("train_", "")
            val_key = f"val_{metric}"

            ax = axes[plot_idx]
            epochs = range(1, len(history[key]) + 1)

            ax.plot(epochs, history[key], label="Train", marker="o", markersize=3)
            if val_key in history:
                ax.plot(epochs, history[val_key], label="Val", marker="s", markersize=3)

            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f'{metric.replace("_", " ").title()} over Training')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_idx += 1

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training curves saved to {save_path}")

    return fig


def plot_keypoint_distribution(
    keypoints: np.ndarray,
    figsize: tuple[int, int] = (10, 10),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot distribution of keypoint positions.

    Args:
        keypoints: Array of keypoints with shape (N, 21, 2) or (N, 21, 3).
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Flatten and plot each keypoint type
    for idx, name in enumerate(HAND_KEYPOINTS):
        pts = keypoints[:, idx, :2]
        color = _get_keypoint_color(idx)
        # Convert BGR to RGB and normalize
        color_rgb = tuple(c / 255 for c in reversed(color))

        ax.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=5, c=[color_rgb], label=name)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Keypoint Distribution")
    ax.invert_yaxis()  # Match image coordinates
    ax.set_aspect("equal")

    # Add legend (grouped by finger)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::4], labels[::4], loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_asl_reference_image(
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (15, 10),
) -> plt.Figure:
    """Create a reference image showing all ASL letters."""
    fig, axes = plt.subplots(5, 6, figsize=figsize)
    axes = axes.flatten()

    for idx, (ax, letter) in enumerate(zip(axes, ASL_CLASSES[:26])):
        ax.text(
            0.5,
            0.5,
            letter,
            fontsize=40,
            ha="center",
            va="center",
            fontweight="bold",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(f"ASL {letter}", fontsize=10)

    # Hide extra axes
    for ax in axes[26:]:
        ax.axis("off")

    plt.suptitle("ASL Alphabet Reference", fontsize=16)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
