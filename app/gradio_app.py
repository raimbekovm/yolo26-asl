"""Gradio web interface for ASL recognition."""

from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np

from src.models.pipeline import ASLPipeline
from src.utils.constants import WEIGHTS_DIR
from src.utils.visualization import draw_hand_keypoints, draw_prediction


# Global pipeline instance
_pipeline: Optional[ASLPipeline] = None


def get_pipeline() -> ASLPipeline:
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ASLPipeline(
            pose_model="yolo26n-pose.pt",
            classifier_model=(
                WEIGHTS_DIR / "asl_classifier.pt"
                if (WEIGHTS_DIR / "asl_classifier.pt").exists()
                else None
            ),
        )
        _pipeline.warmup()
    return _pipeline


def predict_image(image: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Predict ASL letter from uploaded image.

    Args:
        image: Input image (RGB format from Gradio).

    Returns:
        Tuple of (annotated image, prediction text).
    """
    if image is None:
        return None, "Please upload an image"

    pipeline = get_pipeline()

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run prediction
    predictions = pipeline.predict(image_bgr)

    if not predictions:
        return image, "No hand detected. Please show your hand clearly."

    # Annotate image
    annotated = image_bgr.copy()
    results_text = []

    for i, pred in enumerate(predictions):
        # Draw keypoints
        annotated = draw_hand_keypoints(
            annotated,
            pred.keypoints,
            draw_skeleton=True,
        )

        # Draw prediction
        x1 = int(pred.bbox[0])
        y1 = int(pred.bbox[1])
        annotated = draw_prediction(
            annotated,
            pred.letter,
            pred.confidence,
            position=(max(x1, 10), max(y1 - 10, 40)),
        )

        results_text.append(f"Hand {i+1}: **{pred.letter}** ({pred.confidence:.1%} confidence)")

    # Convert back to RGB
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated_rgb, "\n".join(results_text)


def predict_webcam(image: np.ndarray) -> tuple[np.ndarray, str]:
    """Process webcam frame."""
    return predict_image(image)


def create_app() -> gr.Blocks:
    """Create Gradio application."""

    # Custom CSS
    css = """
    .main-title {
        text-align: center;
        margin-bottom: 20px;
    }
    .prediction-box {
        font-size: 24px;
        font-weight: bold;
        padding: 20px;
        text-align: center;
    }
    """

    with gr.Blocks(css=css, title="YOLO26 ASL Recognition") as app:
        gr.Markdown(
            """
            # YOLO26 ASL Recognition

            Real-time American Sign Language alphabet recognition using YOLO26-pose.

            **Features:**
            - 21 hand keypoints detection with YOLO26-pose (NMS-free, 43% faster on CPU)
            - Classification of 26 ASL letters + 5 gestures (Hello, Thank You, Sorry, Yes, No)
            - Real-time webcam support

            ---
            """
        )

        with gr.Tabs():
            # Image Upload Tab
            with gr.TabItem("Upload Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Upload Image",
                            type="numpy",
                            sources=["upload", "clipboard"],
                        )
                        submit_btn = gr.Button("Recognize", variant="primary")

                    with gr.Column():
                        image_output = gr.Image(label="Result")
                        text_output = gr.Markdown(label="Prediction")

                submit_btn.click(
                    fn=predict_image,
                    inputs=[image_input],
                    outputs=[image_output, text_output],
                )

                # Examples
                gr.Examples(
                    examples=(
                        [
                            ["assets/images/example_a.jpg"],
                            ["assets/images/example_b.jpg"],
                            ["assets/images/example_hello.jpg"],
                        ]
                        if Path("assets/images").exists()
                        else []
                    ),
                    inputs=[image_input],
                    outputs=[image_output, text_output],
                    fn=predict_image,
                    cache_examples=False,
                )

            # Webcam Tab
            with gr.TabItem("Webcam"):
                gr.Markdown("**Note:** Webcam access requires HTTPS or localhost.")

                with gr.Row():
                    webcam_input = gr.Image(
                        label="Webcam",
                        type="numpy",
                        sources=["webcam"],
                        streaming=True,
                    )
                    webcam_output = gr.Image(label="Result")

                webcam_text = gr.Markdown("Show ASL letter to the camera")

                webcam_input.stream(
                    fn=predict_webcam,
                    inputs=[webcam_input],
                    outputs=[webcam_output, webcam_text],
                )

            # ASL Reference Tab
            with gr.TabItem("ASL Reference"):
                gr.Markdown(
                    """
                    ## ASL Alphabet Reference

                    | Letter | Description |
                    |--------|-------------|
                    | A | Fist with thumb to the side |
                    | B | Flat hand with fingers together |
                    | C | Curved hand like holding a ball |
                    | ... | ... |

                    ## Supported Gestures

                    - **Hello** - Wave
                    - **Thank You** - Touch chin and move forward
                    - **Sorry** - Fist circling on chest
                    - **Yes** - Fist nodding
                    - **No** - Two fingers closing

                    [Full ASL Alphabet Guide](https://www.handspeak.com/word/asl-abc/)
                    """
                )

            # About Tab
            with gr.TabItem("About"):
                gr.Markdown(
                    """
                    ## About This Project

                    This application demonstrates real-time ASL recognition using:

                    - **YOLO26-pose** - Latest Ultralytics model with NMS-free architecture
                    - **Hand Keypoints** - 21 keypoints per hand (MediaPipe convention)
                    - **MLP Classifier** - Lightweight classifier trained on SignAlphaSet

                    ### Technical Details

                    | Component | Model | Performance |
                    |-----------|-------|-------------|
                    | Hand Detection | YOLO26n-pose | 40ms CPU / 1.8ms GPU |
                    | Keypoints | 21 points | RLE-optimized |
                    | Classifier | MLP (256-128-64) | <1ms |

                    ### Key YOLO26 Features

                    - **NMS-free** - End-to-end architecture without post-processing
                    - **43% faster CPU** - Optimized for edge deployment
                    - **RLE pose** - Residual Log-Likelihood Estimation for accurate keypoints

                    ---

                    **GitHub**: [yolo26-asl](https://github.com/raimbekovm/yolo26-asl)

                    **Built with Ultralytics YOLO26**
                    """
                )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
