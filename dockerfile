# build Leaflet's static assets
FROM node:20-slim AS assets
WORKDIR /build
COPY road_viewer/package.json road_viewer/package-lock.json ./
RUN npm ci

# the actual service REDUCES SIZE BY AN INSANE AMT
FROM python:3.12-slim 
WORKDIR /app

# app deps (psycopg2-binary added here since it's missing from requirements.txt)
COPY road_viewer/requirements.txt ./road_viewer/requirements.txt
RUN pip install --no-cache-dir -r road_viewer/requirements.txt psycopg2-binary

# App code
COPY road_viewer ./road_viewer

# Leaflet's built assets, pulled from stage 1 (only what server.py mounts)
COPY --from=assets /build/node_modules/leaflet/dist ./road_viewer/node_modules/leaflet/dist

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8080

# Shell form so $PORT (injected by Cloud Run) actually expands
CMD uvicorn road_viewer.server:create_app --factory --host 0.0.0.0 --port ${PORT:-8080}