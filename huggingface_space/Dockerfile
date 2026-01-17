FROM ultralytics/ultralytics:latest-cpu

WORKDIR /app

RUN pip install --no-cache-dir gradio>=4.44.0

COPY app.py .
COPY yolo26n_asl.pt .

EXPOSE 7860

CMD ["python", "app.py"]
