"""Test demo view endpoints, seeding, and dynamic pothole ingestion."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starlette.testclient import TestClient
from road_viewer.tiger_db import (
    init_db,
    seed_simulated_detections,
    clear_simulated_detections,
    get_simulated_detections,
    add_pothole,
    get_potholes,
    delete_pothole,
)
from road_viewer.server import create_app


def test_seed_and_quality_values():
    init_db()
    records = seed_simulated_detections(clear_existing=True)
    assert len(records) > 50

    # Verify road_quality values are exclusively 'good', 'medium', or 'bad'
    qualities = set(r["road_quality"] for r in records)
    assert qualities.issubset({"good", "medium", "bad"})
    assert "good" in qualities
    assert "bad" in qualities

    # Check vehicle grouping
    cars = set(r["car_id"] for r in records)
    assert "Car-Alpha (Fleet Patrol)" in cars
    assert "Car-Beta (Transit Bus #12)" in cars
    assert "Car-Gamma (Courier Van)" in cars

    # Normal GPS points must have imu=False, yolo=False, road_quality='good'
    gps_points = [r for r in records if not r["imu"] and not r["yolo"]]
    assert len(gps_points) > 0
    assert all(r["road_quality"] == "good" for r in gps_points)

    # Anomaly points must have road_quality 'medium' or 'bad'
    anomalies = [r for r in records if r["imu"] or r["yolo"]]
    assert len(anomalies) > 0
    assert all(r["road_quality"] in ("medium", "bad") for r in anomalies)
    print(f"[OK] Seeding verified: {len(records)} records, {len(gps_points)} normal GPS pings, {len(anomalies)} anomalies")


def test_demo_view_api_endpoints():
    app = create_app()
    client = TestClient(app)

    # 1. Test /api/simulated-detections/seed
    seed_res = client.post("/api/simulated-detections/seed", json={"clear_existing": True})
    assert seed_res.status_code == 200
    seed_data = seed_res.json()
    assert seed_data["status"] == "ok"
    assert seed_data["count"] > 50

    # 2. Test /api/simulated-detections query
    list_res = client.get("/api/simulated-detections?order=asc&limit=200")
    assert list_res.status_code == 200
    rows = list_res.json()
    assert len(rows) > 0
    assert "road_quality" in rows[0]
    assert rows[0]["road_quality"] in ("good", "medium", "bad")

    # 3. Test /api/simulated-detections/ingest
    ingest_payload = {
        "latitude": 43.4799,
        "longitude": -80.5399,
        "imu": True,
        "yolo": True,
        "car_id": "Test-Car-1",
    }
    ingest_res = client.post("/api/simulated-detections/ingest", json=ingest_payload)
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert ingest_data["status"] in ("created", "already_exists")
    if ingest_data["status"] == "created":
        pothole_id = ingest_data["pothole"]["id"]
        # Cleanup created pothole
        client.delete(f"/api/potholes/{pothole_id}")

    # 4. Test /demo redirect and /demo_view/index.html
    demo_redirect = client.get("/demo", follow_redirects=False)
    assert demo_redirect.status_code in (307, 302, 301)
    assert "/demo_view" in demo_redirect.headers.get("location", "")

    demo_html = client.get("/demo_view/index.html")
    assert demo_html.status_code == 200
    assert "RoadScope" in demo_html.text
    print("[OK] All demo_view API and static routes verified successfully!")


if __name__ == "__main__":
    test_seed_and_quality_values()
    test_demo_view_api_endpoints()
    print("All demo_view tests passed!")
