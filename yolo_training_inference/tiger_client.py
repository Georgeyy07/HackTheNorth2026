"""Reports YOLO pothole/crack detections into TigerDB.

Reuses the road_viewer FastAPI service's existing POST /api/potholes route
(road_viewer/tiger_db.add_pothole under the hood) instead of talking to
Postgres directly, so this stays on Armaan's side of the contract.
"""

import requests

DEFAULT_TIGER_URL = "http://127.0.0.1:8765"


def confidence_to_severity(confidence: float) -> str:
    """Maps a YOLO detection confidence to the severity buckets add_pothole() expects."""
    if confidence >= 0.75:
        return "CRITICAL"
    if confidence >= 0.6:
        return "HIGH"
    if confidence >= 0.4:
        return "MEDIUM"
    return "LOW"


def report_detection(latitude, longitude, confidence, tiger_url=DEFAULT_TIGER_URL, timeout=5):
    """POSTs one detection to TigerDB. Returns the created record, or None on
    failure -- a network/server hiccup shouldn't crash a live detection loop."""
    try:
        resp = requests.post(
            f"{tiger_url}/api/potholes",
            json={
                "latitude": latitude,
                "longitude": longitude,
                "severity": confidence_to_severity(confidence),
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"  [tiger_db] failed to report detection: {exc}")
        return None
