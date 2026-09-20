"""Inference extracted from adding-dockerfile's trained YOLO26 and per-class filter."""
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import threading
from PIL import Image
from .contract import decode_image, filter_detections


class YoloPredictor:
    def __init__(self, weights, device='cpu'):
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        from ultralytics import YOLO
        path = Path(weights)
        manifest = json.loads(Path(__file__).with_name('manifest.json').read_text())
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.sha256 != manifest['checkpoint_sha256']:
            raise ValueError('YOLO checkpoint does not match the imported source manifest')
        self.model = YOLO(str(path))
        self.classes = {int(k): v for k, v in self.model.names.items()}
        if self.classes != {int(k): v for k,v in manifest['classes'].items()}:
            raise ValueError('YOLO class names do not match the manifest')
        self.device, self.lock = device, threading.Lock()

    def predict(self, payload):
        data, width, height = decode_image(payload['image'])
        with Image.open(BytesIO(data)) as im:
            image = im.convert('RGB')
        # Keep the source's lower crack threshold before per-class filtering.
        floor = .2 if 'crack' in self.classes.values() else .4
        with self.lock:
            result = self.model.predict(image, conf=floor, imgsz=640, device=self.device, verbose=False)[0]
        detections = [dict(box=b.xyxy[0].tolist(), conf=float(b.conf[0]), cls=int(b.cls[0])) for b in result.boxes]
        return dict(detections=filter_detections({'detections': detections},width,height,self.classes),
                    stub=False, checkpoint_sha256=self.sha256, classes=self.classes)
