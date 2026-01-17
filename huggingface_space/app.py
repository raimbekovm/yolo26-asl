"""YOLO26 ASL Detection Demo"""
import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("Loading YOLO26 model...")
if os.path.exists("yolo26n_asl.pt"):
    model = YOLO("yolo26n_asl.pt")
    print("Loaded: yolo26n_asl.pt (trained on ASL)")
else:
    model = YOLO("yolo26n.pt")
    print("Loaded: yolo26n.pt (pretrained)")

def detect(image, conf=0.25):
    if image is None:
        return None, "Upload an image"
    results = model.predict(image, conf=conf, verbose=False)[0]
    annotated_bgr = results.plot()
    annotated = annotated_bgr[..., ::-1]  # BGR to RGB
    if results.boxes is not None and len(results.boxes) > 0:
        text = f"**{len(results.boxes)} detections**\n"
        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            cls_name = results.names[cls_id]
            text += f"- **{cls_name}**: {confidence:.1%}\n"
    else:
        text = "No detections. Try lowering confidence."
    return annotated, text

with gr.Blocks(title="YOLO26 ASL Detection") as demo:
    gr.Markdown("""# YOLO26 ASL Letter Detection

| Model | mAP50 | CPU Speed |
|-------|-------|-----------|
| YOLO26n | 0.751 | **122.8ms** |
| YOLO11n | **0.906** | 127.2ms |

*YOLO26 is 3.6% faster on CPU!*
""")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="Input", type="pil")
            conf = gr.Slider(0.01, 1, 0.25, label="Confidence")
            btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            img_out = gr.Image(label="Output")
            txt = gr.Markdown()
    btn.click(detect, [img_in, conf], [img_out, txt])
    gr.Markdown("[GitHub](https://github.com/raimbekovm/yolo26-asl) | [YOLO26 Docs](https://docs.ultralytics.com/models/yolo26/) | Author: Murat Raimbekov")

demo.launch(server_name="0.0.0.0", server_port=7860)
