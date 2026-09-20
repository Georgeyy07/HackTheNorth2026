# a dockerfile can only run one container
# this is for fallback servicing
# both dockerfiles work with complete different libraries
# this just handles yolos image inference
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
# Install CPU-only torch first. This container has no GPU (it's a Cloud Run
# CPU fallback), but pip's default torch wheel on Linux is the CUDA build,
# which drags in ~1GB of unused nvidia-*/triton packages via ultralytics'
# dependency on torch. Pre-installing the CPU build here means the next
# step finds torch already satisfied and skips the GPU resolution entirely.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}