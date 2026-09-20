"""Serves the trained pothole YOLO weights on Baseten.

Expects model_input = {"image": "<base64-encoded image bytes>", "conf": 0.4}
and returns the same {"detections": [{"box", "conf", "cls", "label"}]} shape
yolo_fallback/server.py uses, so callers can treat both the same way.
"""

import base64
from pathlib import Path

import numpy as np


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._model = None

    def load(self):
        from ultralytics import YOLO
        weights_path = Path(self._data_dir) / "best.pt"
        self._model = YOLO(str(weights_path))

    def predict(self, model_input):
        import cv2

        image_b64 = model_input.get("image")
        if not image_b64:
            return {"error": "expected {'image': '<base64-encoded image>'}"}

        image_bytes = base64.b64decode(image_b64)
        data = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "could not decode image"}

        conf = float(model_input.get("conf", 0.4))
        results = self._model(img, conf=conf)[0]

        return {
            "detections": [
                {
                    "box": box.xyxy[0].tolist(),
                    "conf": float(box.conf[0]),
                    "cls": int(box.cls[0]),
                    "label": results.names[int(box.cls[0])],
                }
                for box in results.boxes
            ]
        }
