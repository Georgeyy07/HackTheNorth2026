"""Camera-to-cloud-to-Tiger orchestration without importing YOLO into FastAPI."""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid
from .client import VisionClient
from .contract import decode_image, number
from .store import VisionStore

MANIFEST = json.loads(Path(__file__).with_name('manifest.json').read_text())


class VisionService:
    def __init__(self, client, store):
        self.client, self.store = client, store

    @classmethod
    def from_env(cls):
        if os.environ.get('YOLO_INFERENCE_ENABLED', '').lower() not in ('true','1'):
            return None
        url = next((os.environ[k] for k in ('VISION_DATABASE_URL','DATABASE_URL','TIGER_DATA_URL','IMU_DATABASE_URL','POSTGRES_URL') if os.environ.get(k)), None)
        if not url:
            raise ValueError('Vision inference needs a database URL')
        client = VisionClient(os.environ.get('YOLO_BASETEN_PREDICT_URL'),
                              os.environ.get('YOLO_BASETEN_API_KEY') or os.environ.get('BASETEN_API_KEY'),
                              os.environ.get('YOLO_FALLBACK_URL'),
                              classes={int(k):v for k,v in MANIFEST['classes'].items()},
                              checkpoint_sha256=MANIFEST['checkpoint_sha256'])
        return cls(client, VisionStore(url, publish_potholes=os.environ.get('YOLO_PUBLISH_POTHOLES','true').lower() in ('true','1')))

    async def process(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('Expected a camera frame object')
        frame_id, device_id = payload.get('frame_id'), payload.get('device_id')
        try:
            frame_id = str(uuid.UUID(frame_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError('frame_id must be a UUID retained across retries') from exc
        if not isinstance(device_id, str) or not 1 <= len(device_id) <= 128:
            raise ValueError('device_id is required (1..128 characters)')
        timestamp, frame_number = payload.get('timestamp'), payload.get('frame_number')
        if not number(timestamp) or not 1e12 <= timestamp <= 1e13 or type(frame_number) is not int or frame_number < 0:
            raise ValueError('Expected Unix milliseconds timestamp and nonnegative frame_number')
        data, width, height = decode_image(payload.get('image'))
        lat, lon = payload.get('latitude'), payload.get('longitude')
        gps_time, gps_received = payload.get('gps_timestamp_ms'), payload.get('gps_received_at_ms')
        location_valid = (all(number(v) for v in (lat,lon,gps_time,gps_received)) and -90<=lat<=90 and -180<=lon<=180
                          and 0<=timestamp-gps_time<=3000 and gps_received<=timestamp)
        frame = dict(frame_id=frame_id,device_id=device_id,frame_number=frame_number,timestamp=timestamp,
                     captured_at=datetime.fromtimestamp(timestamp/1000,timezone.utc).isoformat(),
                     image_sha256=hashlib.sha256(data).hexdigest(),
                     latitude=lat if location_valid else None,longitude=lon if location_valid else None)
        # Bind raw location metadata too, so a retry cannot silently change its GPS.
        identity = dict(frame, gps=[lat,lon,gps_time,gps_received])
        try:
            request_sha = hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid GPS metadata') from exc
        cached = await asyncio.to_thread(self.store.lookup,frame_id,request_sha)
        if cached is not None:
            return cached
        result = await self.client.predict(data,width,height)
        output = await asyncio.to_thread(self.store.save,frame,request_sha,result)
        if output['pothole_ids']:
            from alert_service.potholes import invalidate_potholes_cache
            invalidate_potholes_cache()
        return output


def install_routes(app, service):
    from fastapi import HTTPException, Body

    @app.get('/api/vision/status')
    def status():
        return dict(enabled=service is not None, model=MANIFEST,
                    storage=('sqlite' if service.store.sqlite else 'postgresql') if service else None,
                    publish_potholes=service.store.publish_potholes if service else False)

    @app.post('/api/vision/predict')
    async def predict(payload: dict = Body(...)):
        if service is None:
            raise HTTPException(503, 'Vision inference is not configured')
        try:
            return await service.process(payload)
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503,'Vision inference or storage unavailable; retry the same frame_id') from exc

    @app.get('/api/vision/frames/{frame_id}')
    async def frame(frame_id: str):
        if service is None:
            raise HTTPException(503,'Vision inference is not configured')
        result = await asyncio.to_thread(service.store.frame,frame_id)
        if result is None:
            raise HTTPException(404,'Unknown frame')
        return result

    @app.on_event('shutdown')
    async def shutdown():
        if service is not None:
            await service.client.close()
