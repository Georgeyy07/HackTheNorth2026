"""Minimal HTTP surface so a phone can push GPS pings and the service can run standalone.

POST /vehicles/{vehicle_id}/location   - push one GPS/heading/speed reading
GET  /healthz                          - liveness check

Alert delivery back to the phone (for Gemini to speak) isn't wired to a
transport yet -- `log_sink` just logs each triggered alert. Swap it for a
websocket/push sink once the frontend side is ready to receive it.

Run with: uvicorn alert_service.api:app --reload
Pothole storage goes through road_viewer.tiger_db, which reads its own
connection settings from a workspace .env (DATABASE_URL / TIGER_DATA_URL /
POSTGRES_URL) and falls back to a local SQLite file if none connect.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from .alert_math import AlertResult, VehicleState
from .gps import QueueGPSSource
from .potholes import PotholeStore, TigerDBPotholeStore
from .service import AlertService

logger = logging.getLogger(__name__)


class LocationPing(BaseModel):
    lat: float
    lon: float
    heading_deg: float
    speed_mps: float


async def log_sink(vehicle_id: str, result: AlertResult) -> None:
    logger.info("alert vehicle=%s %s", vehicle_id, result)


def create_app(potholes: PotholeStore) -> FastAPI:
    app = FastAPI()
    service = AlertService(potholes, log_sink)
    gps_sources: dict[str, QueueGPSSource] = {}

    @app.post("/vehicles/{vehicle_id}/location")
    async def post_location(vehicle_id: str, ping: LocationPing) -> dict:
        source = gps_sources.get(vehicle_id)
        if source is None:
            source = QueueGPSSource()
            gps_sources[vehicle_id] = source
            service.add_vehicle(vehicle_id, source)
        source.push(
            VehicleState(
                lat=ping.lat,
                lon=ping.lon,
                heading_deg=ping.heading_deg,
                speed_mps=ping.speed_mps,
            )
        )
        return {"ok": True}

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    return app


app = create_app(TigerDBPotholeStore())
