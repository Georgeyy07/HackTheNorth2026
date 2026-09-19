"""Nearby-pothole lookups for the alert service.

George's TigerDB schema isn't built yet. PostgresPotholeStore assumes a
`potholes` table with (id, lat, lon, severity) columns, reached over the
standard Postgres wire protocol -- TigerDB is Postgres-compatible. When the
real schema lands, only the SQL in `query_nearby` needs to change; everything
that calls PotholeStore stays the same.
"""

from __future__ import annotations

import math
from typing import Iterable, List

from .alert_math import PotholeReport, haversine_distance_m

# ~1 degree of latitude is ~111.32km; used to turn a meter radius into a cheap
# bounding-box pre-filter before TigerDB has PostGIS-style distance queries.
METERS_PER_DEGREE_LAT = 111_320.0


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


class PostgresPotholeStore(PotholeStore):
    """Queries TigerDB over asyncpg. Requires the `asyncpg` package."""

    def __init__(self, pool) -> None:
        self._pool = pool  # asyncpg.Pool

    @classmethod
    async def connect(cls, dsn: str) -> "PostgresPotholeStore":
        import asyncpg

        pool = await asyncpg.create_pool(dsn)
        return cls(pool)

    async def query_nearby(self, lat: float, lon: float, radius_m: float) -> List[PotholeReport]:
        lat_delta = radius_m / METERS_PER_DEGREE_LAT
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * max(0.1, math.cos(math.radians(lat)))
        lon_delta = radius_m / meters_per_degree_lon

        rows = await self._pool.fetch(
            """
            SELECT id, lat, lon, severity
            FROM potholes
            WHERE lat BETWEEN $1 AND $2
              AND lon BETWEEN $3 AND $4
            """,
            lat - lat_delta,
            lat + lat_delta,
            lon - lon_delta,
            lon + lon_delta,
        )
        candidates = [
            PotholeReport(id=str(r["id"]), lat=r["lat"], lon=r["lon"], severity=r["severity"])
            for r in rows
        ]
        # The bounding box over-selects near the box corners; trim with the
        # real great-circle distance so `max_alert_distance_m` is exact.
        return [p for p in candidates if haversine_distance_m(lat, lon, p.lat, p.lon) <= radius_m]
