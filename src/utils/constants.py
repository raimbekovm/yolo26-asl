"""Project-wide constants."""

from pathlib import Path
from typing import Final


# ==================== Paths ====================
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent.parent
CONFIG_DIR: Final[Path] = PROJECT_ROOT / "configs"
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
WEIGHTS_DIR: Final[Path] = PROJECT_ROOT / "weights"
OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"

# ==================== ASL Classes ====================
ASL_LETTERS: Final[list[str]] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]

ASL_GESTURES: Final[list[str]] = [
    "Hello", "ThankYou", "Sorry", "Yes", "No",
]

ASL_CLASSES: Final[list[str]] = ASL_LETTERS + ASL_GESTURES
NUM_CLASSES: Final[int] = len(ASL_CLASSES)

# Class to index mapping
CLASS_TO_IDX: Final[dict[str, int]] = {cls: idx for idx, cls in enumerate(ASL_CLASSES)}
IDX_TO_CLASS: Final[dict[int, str]] = {idx: cls for idx, cls in enumerate(ASL_CLASSES)}

# ==================== Hand Keypoints ====================
NUM_KEYPOINTS: Final[int] = 21
KEYPOINT_DIMS: Final[int] = 3  # x, y, visibility

HAND_KEYPOINTS: Final[list[str]] = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# Skeleton connections for visualization (pairs of keypoint indices)
HAND_SKELETON: Final[list[tuple[int, int]]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17),
]

# Flip indices for horizontal augmentation
FLIP_IDX: Final[list[int]] = list(range(21))  # Hand keypoints don't need flipping

# ==================== Colors (BGR for OpenCV) ====================
COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "orange": (0, 165, 255),
}

# Keypoint colors by finger
FINGER_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "thumb": (0, 255, 0),      # Green
    "index": (255, 0, 0),      # Blue
    "middle": (0, 255, 255),   # Yellow
    "ring": (0, 165, 255),     # Orange
    "pinky": (255, 0, 255),    # Magenta
    "wrist": (255, 255, 255),  # White
}

# ==================== Model Defaults ====================
DEFAULT_POSE_MODEL: Final[str] = "yolo26n-pose.pt"
DEFAULT_CLASSIFIER_MODEL: Final[str] = "asl_classifier.pt"
DEFAULT_IMAGE_SIZE: Final[int] = 640
DEFAULT_CONF_THRESHOLD: Final[float] = 0.5
DEFAULT_IOU_THRESHOLD: Final[float] = 0.7

# ==================== Dataset URLs ====================
DATASET_URLS: Final[dict[str, str]] = {
    "signalphaset": "https://data.mendeley.com/datasets/8fmvr9m98w/3",
    "hand_keypoints": "https://github.com/ultralytics/assets/releases/download/v0.0.0/hand-keypoints.zip",
}
