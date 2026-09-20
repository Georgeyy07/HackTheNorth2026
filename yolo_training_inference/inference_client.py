"""Runs pothole/crack detection against Baseten first, failing over to the
self-hosted Cloud Run service (yolo_fallback) if Baseten doesn't answer
within BASETEN_TIMEOUT_S -- whether that's a cold start, an outage, or just
a slow response.

The two backends have different request/response shapes (Baseten's model.py
takes JSON + base64, the Cloud Run fallback takes multipart form-data), so
this normalizes both into the same call.
"""

import base64
import os

import requests

BASETEN_URL = "https://model-q86o5l53.api.baseten.co/environments/production/predict"
FALLBACK_URL = "https://pothole-yolo-fallback-148999688329.northamerica-northeast2.run.app/predict"

# Baseten's production deployment scales to zero (min_replica=0), so a cold
# start can take much longer than this -- that's intentional: if Baseten
# isn't warm and responding fast, we'd rather serve from the always-on
# Cloud Run fallback than block a live camera-frame pipeline waiting for it.
BASETEN_TIMEOUT_S = 4.0
FALLBACK_TIMEOUT_S = 10.0


def predict(image_bytes, conf=0.4, baseten_api_key=None, baseten_timeout=BASETEN_TIMEOUT_S):
    """Returns {"source": "baseten"|"fallback", "detections": [...]}."""
    api_key = baseten_api_key or os.environ.get("BASETEN_API_KEY")
    if api_key:
        try:
            resp = requests.post(
                BASETEN_URL,
                json={"image": base64.b64encode(image_bytes).decode(), "conf": conf},
                headers={"Authorization": f"Api-Key {api_key}"},
                timeout=baseten_timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            if "error" not in result:
                return {"source": "baseten", **result}
        except (requests.RequestException, ValueError):
            pass  # timed out, unreachable, or bad response -- fail over below

    resp = requests.post(
        FALLBACK_URL,
        files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
        timeout=FALLBACK_TIMEOUT_S,
    )
    resp.raise_for_status()
    return {"source": "fallback", **resp.json()}
