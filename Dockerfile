# Adapted from adding-dockerfile: the combined backend includes both inference clients.
FROM node:20-slim AS assets
WORKDIR /build
COPY road_viewer/package.json road_viewer/package-lock.json ./
RUN npm ci

FROM python:3.11-slim
WORKDIR /app
COPY road_viewer/requirements*.txt /app/road_viewer/
RUN pip install --no-cache-dir -r road_viewer/requirements-backend.txt
COPY road_viewer /app/road_viewer
COPY alert_service /app/alert_service
COPY route_planner /app/route_planner
COPY imu_inference /app/imu_inference
COPY vision_inference /app/vision_inference
COPY --from=assets /build/node_modules/leaflet/dist /app/road_viewer/node_modules/leaflet/dist
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
EXPOSE 8080
CMD uvicorn road_viewer.server:create_app --factory --host 0.0.0.0 --port ${PORT:-8080}
