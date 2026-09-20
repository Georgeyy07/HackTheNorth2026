import asyncio
import sqlite3

import pytest

from road_viewer import tiger_db

from alert_service.potholes import TigerDBPotholeStore


@pytest.fixture
def isolated_tiger_db(tmp_path, monkeypatch):
    """Points road_viewer.tiger_db at a throwaway SQLite file instead of the
    shared artifacts/tiger_potholes.db, so this test can't see or pollute
    data from manual runs or other tests."""
    db_path = tmp_path / "test_potholes.db"

    def fake_connection():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(tiger_db, "get_db_connection", fake_connection)
    monkeypatch.setattr(tiger_db, "USE_POSTGRES", False)
    tiger_db.init_db()
    return tiger_db


def test_query_nearby_maps_columns_and_severity_and_filters_by_radius(isolated_tiger_db):
    isolated_tiger_db.add_pothole(latitude=43.4723, longitude=-80.5430, severity="CRITICAL")
    isolated_tiger_db.add_pothole(latitude=43.60, longitude=-80.10, severity="LOW")

    store = TigerDBPotholeStore()
    results = asyncio.run(store.query_nearby(43.4723, -80.5449, radius_m=400.0))

    assert len(results) == 1
    pothole = results[0]
    assert pothole.lat == pytest.approx(43.4723)
    assert pothole.lon == pytest.approx(-80.5430)
    assert pothole.severity == 1.0  # CRITICAL


def test_query_nearby_defaults_unknown_severity_to_medium_score(isolated_tiger_db):
    isolated_tiger_db.add_pothole(latitude=43.4723, longitude=-80.5430, severity="")

    store = TigerDBPotholeStore()
    results = asyncio.run(store.query_nearby(43.4723, -80.5449, radius_m=400.0))

    assert results[0].severity == 0.5
