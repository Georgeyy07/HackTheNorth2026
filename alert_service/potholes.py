"""Nearby-pothole lookups for the alert service.

TigerDBPotholeStore wraps Armaan's `road_viewer.tiger_db` module (see that
file for the real schema: a `potholes` table with latitude/longitude columns
and a string severity enum, backed by TigerDB/Postgres with a local SQLite
fallback). That module is synchronous psycopg2/sqlite3, so calls run in a
thread via `asyncio.to_thread` to keep this store's async interface. It has
no lat/lon radius filtering built in, so `query_nearby` fetches everything
and filters with `haversine_distance_m` -- fine at hackathon data volumes;
revisit if the potholes table grows large enough to need a DB-side filter.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, List

from .alert_math import PotholeReport, haversine_distance_m

# tiger_db stores severity as a string enum, not the 0-1 float PotholeReport
# expects; this is our own choice of mapping, not part of the DB contract.
SEVERITY_TO_SCORE = {
    "LOW": 0.25,
    "MEDIUM": 0.5,
    "HIGH": 0.75,
    "CRITICAL": 1.0,
}


class PotholeStore:
    async def query_nearby(self, lat: float, lon: float, radius_m: float) -> List[PotholeReport]:
        raise NotImplementedError


class InMemoryPotholeStore(PotholeStore):
    """Backed by a fixed list. For tests and local development without a DB."""

    def __init__(self, potholes: Iterable[PotholeReport]) -> None:
        self._potholes = list(potholes)

    async def query_nearby(self, lat: float, lon: float, radius_m: float) -> List[PotholeReport]:
        return [
            p for p in self._potholes
            if haversine_distance_m(lat, lon, p.lat, p.lon) <= radius_m
        ]


class TigerDBPotholeStore(PotholeStore):
    """Queries potholes via road_viewer.tiger_db, filtering by radius in Python."""

    async def query_nearby(self, lat: float, lon: float, radius_m: float) -> List[PotholeReport]:
        from road_viewer.tiger_db import get_potholes

        rows = await asyncio.to_thread(get_potholes)
        potholes = [
            PotholeReport(
                id=str(row["id"]),
                lat=row["latitude"],
                lon=row["longitude"],
                severity=SEVERITY_TO_SCORE.get(row.get("severity", "MEDIUM"), 0.5),
            )
            for row in rows
        ]
        return [p for p in potholes if haversine_distance_m(lat, lon, p.lat, p.lon) <= radius_m]
