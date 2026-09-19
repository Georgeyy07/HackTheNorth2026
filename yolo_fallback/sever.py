import os
from pathlib import Path
from fastapi import FastAPI, UploadFile
import numpy as np

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
    data = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
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