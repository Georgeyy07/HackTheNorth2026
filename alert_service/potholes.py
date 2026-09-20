"""Nearby-pothole lookups for the alert service.

TigerDBPotholeStore reads from the same TigerDB/Postgres instance as
`road_viewer.tiger_db` (reusing its connection setup), but queries the
`potholes` table directly rather than through `get_potholes()`. That
function's SQL is pinned to an older schema (id, latitude, longitude,
severity, timestamp); the live table has since been redesigned around
multi-device reconciliation (severity_score 0-10, confidence, hit_count,
detected_by_vision/imu, is_active, device_id) -- almost certainly for the
"reconcile conflicting severity reports across cars" work. Since that
redesign only affects the real Postgres table (the local SQLite dev
fallback in tiger_db.py still uses the old simple schema), this module
branches on `USE_POSTGRES` to query the right columns for whichever one
`get_db_connection()` actually connected to.

Reconciling multiple devices' reports of the same physical pothole (the
`hit_count`/`confidence`/`detected_by_*` columns exist for this) is its own
task, not solved here: every active row is currently treated as an
independent pothole for alerting/routing purposes, which double-counts a
pothole that's been reported by several devices rather than merging them.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, List

from .alert_math import PotholeReport, haversine_distance_m

import time

# The local SQLite fallback (road_viewer.tiger_db's dev-only path) still uses
# this older string-enum severity; the live Postgres table uses a 0-10
# severity_score instead (divided by 10 to fit PotholeReport's 0-1 scale).
SEVERITY_TO_SCORE = {
    "LOW": 0.25,
    "MEDIUM": 0.5,
    "HIGH": 0.75,
    "CRITICAL": 1.0,
}


def severity_label(score: float) -> str:
    """Inverse of SEVERITY_TO_SCORE, for displaying a 0-1 score back as LOW/MEDIUM/HIGH/CRITICAL."""
    if score >= 1.0:
        return "CRITICAL"
    if score >= 0.75:
        return "HIGH"
    if score >= 0.5:
        return "MEDIUM"
    return "LOW"


# Candidate names the live Postgres severity column has actually gone by
# this session (it's been renamed twice already -- someone else is actively
# iterating on this schema). Checked against information_schema each call so
# a future rename doesn't require another code change here, just adding the
# new name to this list.
POSTGRES_SEVERITY_COLUMN_CANDIDATES = ["severity", "severity_score", "severity_value"]


def _find_postgres_severity_column(cursor) -> str:
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'potholes'")
    existing = {row[0] for row in cursor.fetchall()}
    for candidate in POSTGRES_SEVERITY_COLUMN_CANDIDATES:
        if candidate in existing:
            return candidate
    raise RuntimeError(
        f"None of {POSTGRES_SEVERITY_COLUMN_CANDIDATES} found in potholes table; "
        f"actual columns: {sorted(existing)}. The live schema changed again -- add the new name."
    )

_POTHOLES_CACHE = None
_POTHOLES_CACHE_TIME = 0.0


def invalidate_potholes_cache():
    global _POTHOLES_CACHE
    _POTHOLES_CACHE = None


def fetch_active_potholes(force_refresh: bool = False) -> List[PotholeReport]:
    """Synchronous fetch of every currently-active pothole report. Branches on
    whichever backend road_viewer.tiger_db actually connected to. Cached for 30s to avoid
    re-establishing cloud DB SSL connections on every route calculation."""
    global _POTHOLES_CACHE, _POTHOLES_CACHE_TIME
    now = time.time()
    if not force_refresh and _POTHOLES_CACHE is not None and (now - _POTHOLES_CACHE_TIME) < 30.0:
        return _POTHOLES_CACHE

    from road_viewer import tiger_db

    conn = tiger_db.get_db_connection()
    cursor = conn.cursor()
    try:
        if tiger_db.USE_POSTGRES:
            severity_column = _find_postgres_severity_column(cursor)
            cursor.execute(f"SELECT id, latitude, longitude, {severity_column} FROM potholes WHERE is_active = true")
            rows = cursor.fetchall()
            # Assumes a 0-10 scale (true for every name this column has had so far).
            result = [
                PotholeReport(id=str(pid), lat=lat, lon=lon, severity=min(1.0, max(0.0, score / 10.0)))
                for pid, lat, lon, score in rows
            ]
        else:
            cursor.execute("SELECT id, latitude, longitude, severity FROM potholes")
            rows = cursor.fetchall()
            result = [
                PotholeReport(id=str(pid), lat=lat, lon=lon, severity=SEVERITY_TO_SCORE.get(severity, 0.5))
                for pid, lat, lon, severity in rows
            ]
        _POTHOLES_CACHE = result
        _POTHOLES_CACHE_TIME = now
        return result
    finally:
        cursor.close()
        conn.close()


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
    """Queries potholes directly from TigerDB, filtering by radius in Python."""

    async def query_nearby(self, lat: float, lon: float, radius_m: float) -> List[PotholeReport]:
        potholes = await asyncio.to_thread(fetch_active_potholes)
        return [p for p in potholes if haversine_distance_m(lat, lon, p.lat, p.lon) <= radius_m]
