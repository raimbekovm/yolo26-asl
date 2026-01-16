"""
YOLO26-ASL: Real-time American Sign Language Recognition.

This package provides end-to-end ASL recognition using YOLO26-pose
for hand keypoint detection and a classifier for letter recognition.

Example:
    >>> from src.inference import ASLPredictor
    >>> predictor = ASLPredictor()
    >>> result = predictor.predict("path/to/image.jpg")
    >>> print(result.letter, result.confidence)
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "Apache-2.0"

from src.utils.logger import setup_logger

# Setup default logger
logger = setup_logger(__name__)
