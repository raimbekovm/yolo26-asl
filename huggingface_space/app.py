"""
YOLO26-ASL HuggingFace Spaces Application
Real-time American Sign Language Recognition
"""

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ASL Classes
ASL_LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
ASL_GESTURES = ['Hello', 'ThankYou', 'Sorry', 'Yes', 'No']
ASL_CLASSES = ASL_LETTERS + ASL_GESTURES
NUM_CLASSES = len(ASL_CLASSES)
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(ASL_CLASSES)}


class ASLClassifierMLP(nn.Module):
    """MLP classifier for ASL recognition."""

    def __init__(self, input_dim=63, num_classes=31, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# Global models
pose_model = None
classifier = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_models():
    """Load YOLO26-pose and classifier models."""
    global pose_model, classifier

    if pose_model is None:
        from ultralytics import YOLO
        pose_model = YOLO('yolo26n-pose.pt')
        print("Loaded YOLO26-pose model")

    if classifier is None:
        classifier = ASLClassifierMLP(input_dim=63, num_classes=NUM_CLASSES)

        # Try to load pretrained weights
        weights_path = Path('asl_classifier.pt')
        if weights_path.exists():
            checkpoint = torch.load(weights_path, map_location=device)
            classifier.load_state_dict(checkpoint['state_dict'])
            print("Loaded classifier weights")
        else:
            print("Warning: No classifier weights found, using random initialization")

        classifier.to(device)
        classifier.eval()

    return pose_model, classifier


def draw_keypoints(image, keypoints, confidence=None):
    """Draw hand keypoints on image."""
    image = image.copy()

    # Skeleton connections
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17),  # Palm
    ]

    kpts = keypoints[:, :2].astype(int)
    conf = confidence if confidence is not None else np.ones(21)

    # Draw skeleton
    for i, j in skeleton:
        if conf[i] > 0.5 and conf[j] > 0.5:
            cv2.line(image, tuple(kpts[i]), tuple(kpts[j]), (0, 255, 255), 2)

    # Draw keypoints
    for idx, (pt, c) in enumerate(zip(kpts, conf)):
        if c > 0.5:
            cv2.circle(image, tuple(pt), 5, (0, 255, 0), -1)
            cv2.circle(image, tuple(pt), 5, (0, 0, 0), 1)

    return image


def predict(image):
    """Run ASL prediction on image."""
    if image is None:
        return None, "Please upload an image"

    # Load models
    pose_model, classifier = load_models()

    # Convert RGB to BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Detect hand keypoints
    results = pose_model(image_bgr, verbose=False)

    if results[0].keypoints is None or results[0].keypoints.xy.shape[0] == 0:
        return image, "No hand detected. Please show your hand clearly."

    # Get keypoints
    kpts = results[0].keypoints.xy[0].cpu().numpy()
    if results[0].keypoints.conf is not None:
        conf = results[0].keypoints.conf[0].cpu().numpy()
    else:
        conf = np.ones(21)

    # Normalize
    h, w = image.shape[:2]
    kpts_norm = kpts.copy()
    kpts_norm[:, 0] /= w
    kpts_norm[:, 1] /= h

    # Classify
    kpts_with_conf = np.column_stack([kpts_norm, conf])
    x = torch.FloatTensor(kpts_with_conf.flatten()).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)

    letter = IDX_TO_CLASS[pred_idx.item()]
    conf_value = confidence.item()

    # Draw on image
    annotated = draw_keypoints(image_bgr, kpts, conf)

    # Add prediction text
    text = f"{letter}: {conf_value:.1%}"
    cv2.putText(annotated, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 0), 4)
    cv2.putText(annotated, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), 2)

    # Convert back to RGB
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    result_text = f"**Predicted: {letter}** ({conf_value:.1%} confidence)"

    return annotated_rgb, result_text


# Create Gradio interface
with gr.Blocks(title="YOLO26 ASL Recognition") as demo:
    gr.Markdown(
        """
        # YOLO26 ASL Recognition ✋

        Real-time American Sign Language alphabet recognition using YOLO26-pose.

        **Features:**
        - NMS-free end-to-end architecture
        - 43% faster CPU inference
        - 21 hand keypoints detection
        - 26 ASL letters + 5 gestures
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="numpy")
            submit_btn = gr.Button("Recognize", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Result")
            output_text = gr.Markdown(label="Prediction")

    submit_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[output_image, output_text]
    )

    gr.Markdown(
        """
        ---
        **Links:** [GitHub](https://github.com/raimbekovm/yolo26-asl) |
        [YOLO26 Docs](https://docs.ultralytics.com/models/yolo26/)

        **Author:** Murat Raimbekov
        """
    )


if __name__ == "__main__":
    demo.launch()
