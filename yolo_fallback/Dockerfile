# a dockerfile can only run one container
# this is for fallback servicing
# both dockerfiles work with complete different libraries
# this just handles yolos image inference
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}