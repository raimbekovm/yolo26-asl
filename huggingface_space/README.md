---
title: YOLO26 ASL Detection
emoji: "\U0001F44B"
colorFrom: purple
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
license: apache-2.0
short_description: ASL letter detection benchmark - YOLO26 vs YOLO11
---

# YOLO26 ASL Letter Detection

Real-time American Sign Language (A-Z) detection using **YOLO26**.

## Benchmark Results

| Model | mAP50 | mAP50-95 | GPU Speed | CPU Speed |
|-------|-------|----------|-----------|-----------|
| YOLO26n | 0.751 | 0.715 | 13.1ms | **122.8ms** |
| YOLO11n | **0.906** | **0.860** | **11.6ms** | 127.2ms |

**Key Finding:** YOLO26 is 3.6% faster on CPU - great for edge deployment!

## Links

- [GitHub Repository](https://github.com/raimbekovm/yolo26-asl)
- [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [Kaggle Notebook](https://www.kaggle.com/code/muraraimbekov/yolo26-vs-yolo11-asl-benchmark)

**Author:** Murat Raimbekov
