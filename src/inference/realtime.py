"""Real-time ASL recognition from webcam."""

import time
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from src.data.dataset import RealtimeKeypointBuffer
from src.models.pipeline import ASLPipeline
from src.utils.constants import ASL_CLASSES, IDX_TO_CLASS
from src.utils.visualization import draw_fps, draw_hand_keypoints, draw_prediction


class RealtimeASL:
    """
    Real-time ASL recognition from webcam.

    Features:
    - Temporal smoothing for stable predictions
    - FPS display
    - Keypoint visualization

    Example:
        >>> realtime = RealtimeASL()
        >>> realtime.run()  # Opens webcam window
    """

    def __init__(
        self,
        pose_model: str = "yolo26n-pose.pt",
        classifier_model: Optional[str] = None,
        device: Optional[str] = None,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
        smoothing_window: int = 5,
        stability_frames: int = 3,
    ):
        """
        Initialize real-time recognizer.

        Args:
            pose_model: Path to pose model.
            classifier_model: Path to classifier model.
            device: Device to run on.
            camera_id: Camera device ID.
            width: Camera width.
            height: Camera height.
            smoothing_window: Window size for temporal smoothing.
            stability_frames: Consecutive frames for stable prediction.
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height

        # Initialize pipeline
        self.pipeline = ASLPipeline(
            pose_model=pose_model,
            classifier_model=classifier_model,
            device=device,
        )

        # Prediction buffer for smoothing
        self.buffer = RealtimeKeypointBuffer(window_size=smoothing_window)
        self.stability_frames = stability_frames

        # State
        self.current_letter = None
        self.current_confidence = 0.0
        self.fps = 0.0

    def run(
        self,
        show_keypoints: bool = True,
        show_skeleton: bool = True,
        show_fps: bool = True,
        mirror: bool = True,
    ) -> None:
        """
        Run real-time recognition.

        Args:
            show_keypoints: Display keypoint overlay.
            show_skeleton: Display skeleton connections.
            show_fps: Display FPS counter.
            mirror: Mirror the camera feed.
        """
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            logger.error(f"Cannot open camera {self.camera_id}")
            return

        logger.info("Starting real-time ASL recognition. Press 'q' to quit.")

        # Warmup
        self.pipeline.warmup()

        frame_times = []

        try:
            while True:
                start_time = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                # Mirror if requested
                if mirror:
                    frame = cv2.flip(frame, 1)

                # Run prediction
                predictions = self.pipeline.predict(frame, max_hands=1)

                # Update buffer
                if predictions:
                    pred = predictions[0]
                    self.buffer.add(
                        pred.keypoints,
                        pred.class_idx,
                        pred.confidence,
                    )

                    # Draw keypoints
                    if show_keypoints or show_skeleton:
                        frame = draw_hand_keypoints(
                            frame,
                            pred.keypoints,
                            draw_skeleton=show_skeleton,
                        )

                # Get smoothed prediction
                if self.buffer.is_stable(self.stability_frames):
                    idx, conf = self.buffer.get_smoothed_prediction()
                    self.current_letter = IDX_TO_CLASS.get(idx, "?")
                    self.current_confidence = conf
                elif not predictions:
                    self.current_letter = None
                    self.current_confidence = 0.0

                # Draw prediction
                if self.current_letter:
                    frame = draw_prediction(
                        frame,
                        self.current_letter,
                        self.current_confidence,
                        position=(50, 100),
                        font_scale=3.0,
                        thickness=3,
                    )

                # Calculate FPS
                frame_time = time.perf_counter() - start_time
                frame_times.append(frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                self.fps = 1.0 / np.mean(frame_times)

                # Draw FPS
                if show_fps:
                    frame = draw_fps(frame, self.fps)

                # Display
                cv2.imshow("ASL Recognition - Press Q to quit", frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        logger.info("Real-time recognition stopped")

    def get_prediction(self) -> tuple[Optional[str], float]:
        """Get current prediction."""
        return self.current_letter, self.current_confidence


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Real-time ASL Recognition")
    parser.add_argument("--pose-model", default="yolo26n-pose.pt")
    parser.add_argument("--classifier", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--no-fps", action="store_true")

    args = parser.parse_args()

    realtime = RealtimeASL(
        pose_model=args.pose_model,
        classifier_model=args.classifier,
        device=args.device,
        camera_id=args.camera,
        width=args.width,
        height=args.height,
    )

    realtime.run(
        mirror=not args.no_mirror,
        show_fps=not args.no_fps,
    )


if __name__ == "__main__":
    main()
