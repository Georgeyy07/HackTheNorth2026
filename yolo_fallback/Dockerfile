# Build from repository root: docker build -f yolo_fallback/Dockerfile .
FROM python:3.11-slim
WORKDIR /app
COPY yolo_fallback/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch==2.9.0+cpu torchvision==0.24.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt && pip uninstall -y opencv-python && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.10.0.84
COPY vision_inference /app/vision_inference
COPY yolo_fallback /app/yolo_fallback
COPY models/yolo26/best.pt /app/models/yolo26/best.pt
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD uvicorn yolo_fallback.server:create_app --factory --host 0.0.0.0 --port ${PORT:-8080}
