"""Test simulated_detections table in Tiger Data database and REST API endpoints."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starlette.testclient import TestClient
from road_viewer.tiger_db import (
    init_db,
    add_simulated_detection,
    get_simulated_detections,
    get_simulated_detection_by_id,
    delete_simulated_detection,
)
from road_viewer.server import create_app


def test_simulated_detections_crud():
    # 1. Initialize table
    init_db()

    ts = 1726800000123
    test_car = "test-vehicle-42"

    # 2. Add detection records
    d1 = add_simulated_detection(
        timestamp=ts,
        imu=True,
        yolo=False,
        latitude=43.4723,
        longitude=-80.5449,
        car_id=test_car,
    )
    assert d1["id"] is not None
    assert d1["timestamp"] == ts
    assert d1["imu"] is True
    assert d1["yolo"] is False
    assert abs(d1["latitude"] - 43.4723) < 1e-4
    assert abs(d1["longitude"] - (-80.5449)) < 1e-4
    assert d1["car_id"] == test_car
    print("[OK] add_simulated_detection created record #", d1["id"])

    d2 = add_simulated_detection(
        timestamp=ts + 500,
        imu=False,
        yolo=True,
        latitude=43.4750,
        longitude=-80.5380,
        car_id=test_car,
    )
    assert d2["imu"] is False
    assert d2["yolo"] is True

    d3 = add_simulated_detection(
        timestamp=ts + 1000,
        imu=True,
        yolo=True,
        latitude=43.4680,
        longitude=-80.5250,
        car_id="other-car-99",
    )
    assert d3["imu"] is True
    assert d3["yolo"] is True

    # 3. Query all and filter
    all_car_records = get_simulated_detections(car_id=test_car)
    assert len(all_car_records) >= 2
    for r in all_car_records:
        assert r["car_id"] == test_car
        assert isinstance(r["timestamp"], int)
        assert isinstance(r["imu"], bool)
        assert isinstance(r["yolo"], bool)
        assert isinstance(r["latitude"], float)
        assert isinstance(r["longitude"], float)
    print(f"[OK] get_simulated_detections returned {len(all_car_records)} records for car {test_car}")

    # 4. Filter by imu / yolo
    imu_true = get_simulated_detections(car_id=test_car, imu=True)
    assert any(r["id"] == d1["id"] for r in imu_true)
    assert not any(r["id"] == d2["id"] for r in imu_true)

    yolo_true = get_simulated_detections(car_id=test_car, yolo=True)
    assert any(r["id"] == d2["id"] for r in yolo_true)
    assert not any(r["id"] == d1["id"] for r in yolo_true)
    print("[OK] Boolean filtering by imu and yolo verified")

    # 5. Get by ID
    fetched = get_simulated_detection_by_id(d1["id"])
    assert fetched is not None
    assert fetched["id"] == d1["id"]
    assert fetched["car_id"] == test_car
    assert fetched["imu"] is True

    # 6. Delete
    deleted = delete_simulated_detection(d1["id"])
    assert deleted is True
    assert get_simulated_detection_by_id(d1["id"]) is None
    delete_simulated_detection(d2["id"])
    delete_simulated_detection(d3["id"])
    print("[OK] delete_simulated_detection verified cleanly")


def test_simulated_detections_api():
    app = create_app()
    client = TestClient(app)

    # 1. POST /api/simulated-detections
    payload = {
        "timestamp": int(time.time() * 1000),
        "imu": True,
        "yolo": True,
        "latitude": 43.4730,
        "longitude": -80.5420,
        "car_id": "api-test-car",
    }
    resp = client.post("/api/simulated-detections", json=payload)
    assert resp.status_code == 200, resp.text
    created = resp.json()
    assert created["id"] is not None
    assert created["imu"] is True
    assert created["yolo"] is True
    assert created["car_id"] == "api-test-car"
    record_id = created["id"]
    print(f"[OK] POST /api/simulated-detections created record #{record_id}")

    # 2. POST /api/simulated-detections with 'lattitude' typo tolerance
    payload_typo = {
        "timestamp": int(time.time() * 1000),
        "imu": False,
        "yolo": True,
        "lattitude": 43.4800,
        "longitude": -80.5300,
        "car_id": "api-test-car",
    }
    resp_typo = client.post("/api/simulated-detections", json=payload_typo)
    assert resp_typo.status_code == 200, resp_typo.text
    created_typo = resp_typo.json()
    assert abs(created_typo["latitude"] - 43.4800) < 1e-4
    print("[OK] POST /api/simulated-detections supported 'lattitude' payload spelling")

    # 3. GET /api/simulated-detections
    get_resp = client.get("/api/simulated-detections?car_id=api-test-car")
    assert get_resp.status_code == 200, get_resp.text
    records = get_resp.json()
    assert len(records) >= 2
    assert all(r["car_id"] == "api-test-car" for r in records)
    print(f"[OK] GET /api/simulated-detections verified with {len(records)} records")

    # 4. GET /api/simulated-detections/{id}
    fetch_resp = client.get(f"/api/simulated-detections/{record_id}")
    assert fetch_resp.status_code == 200, fetch_resp.text
    assert fetch_resp.json()["id"] == record_id

    # 5. DELETE /api/simulated-detections/{id}
    del_resp = client.delete(f"/api/simulated-detections/{record_id}")
    assert del_resp.status_code == 200, del_resp.text
    client.delete(f"/api/simulated-detections/{created_typo['id']}")
    print("[OK] DELETE /api/simulated-detections/{id} verified")

    # 6. Verify alias /api/detections also works
    alias_resp = client.post("/api/detections", json=payload)
    assert alias_resp.status_code == 200, alias_resp.text
    alias_id = alias_resp.json()["id"]
    client.delete(f"/api/detections/{alias_id}")
    print("[OK] /api/detections alias verified")


if __name__ == "__main__":
    test_simulated_detections_crud()
    test_simulated_detections_api()
    print("All simulated_detections table & API tests passed successfully!")
