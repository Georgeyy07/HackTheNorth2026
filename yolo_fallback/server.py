import os
from pathlib import Path

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from fastapi import FastAPI, UploadFile
import numpy as np

# dsn=None (SENTRY_DSN unset) makes the SDK a safe no-op instead of erroring,
# so this is always safe to leave in -- same pattern as road_viewer/server.py.
sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
)

app = FastAPI()

MODEL_PATH = Path(__file__).parent / "best.pt"
model = None
if MODEL_PATH.exists():
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()

    if model is None:
        # Stub mode: no trained weights yet. Returns a fake detection so the
        # Cloud Run -> fallback -> caller pipeline can be tested end-to-end
        # before Armaan's model exists.
        return {
            "detections": [
                {"box": [100.0, 120.0, 240.0, 260.0], "conf": 0.42, "cls": 0}
            ],
            "stub": True,
        }

    import cv2
    with sentry_sdk.start_span(op="inference", name="yolo_fallback.predict"):
        data = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            sentry_sdk.capture_message(
                f"yolo_fallback: cv2.imdecode failed for upload "
                f"'{file.filename}' ({len(contents)} bytes)",
                level="error",
            )
            return {"detections": [], "stub": False, "error": "could not decode image"}
        results = model(img)[0]
    return {
        "detections": [
            {"box": b.xyxy[0].tolist(), "conf": float(b.conf[0]), "cls": int(b.cls[0])}
            for b in results.boxes
        ],
        "stub": False,
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
