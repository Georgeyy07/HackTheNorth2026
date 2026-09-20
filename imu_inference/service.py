"""Transactional per-connection inference sessions and FastAPI integration."""
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid
from .client import BasetenClient, ENSEMBLE_SHA
from .filters import DEFAULT_ALERT_FILTER
from .store import PredictionStore
from .stream import RoadStream


class InferenceSession:
    def __init__(self, service, metadata):
        self.service = service
        self.id = str(uuid.uuid4())
        self.stream = RoadStream(service.calibration)
        self.last_batch = -1
        self.last_digest = self.last_response = None
        self.total_samples = 0
        self.metadata = dict(metadata, ensemble_sha256=ENSEMBLE_SHA, alert_filter=DEFAULT_ALERT_FILTER,
                             calibration=service.calibration)

    async def process(self, payload):
        batch_id, samples = payload.get('batch_id'), payload.get('samples')
        if type(batch_id) is not int or batch_id < 0 or not isinstance(samples, list) or not 1 <= len(samples) <= 1024:
            raise ValueError('Expected nonnegative integer batch_id and 1..1024 samples')
        if any(not isinstance(s, dict) for s in samples):
            raise ValueError('Each sample must be an object')
        digest = hashlib.sha256(json.dumps(samples, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if batch_id == self.last_batch:
            if digest != self.last_digest:
                raise ValueError('Batch ID reused with different samples')
            return self.last_response
        if batch_id < self.last_batch:
            raise ValueError('Batch IDs must increase')
        # Neither cloud errors nor failed DB commits advance consensus or Kalman state.
        candidate = copy.deepcopy(self.stream)
        updates = await candidate.push(samples, self.service.client)
        for row in updates:
            row['session_id'] = self.id
        persisted = await asyncio.to_thread(self.service.store.save, self.id, updates)
        response = dict(status='ok', session_id=self.id, batch_id=batch_id,
                        received_samples=len(samples), total_samples=self.total_samples+len(samples),
                        persisted=persisted, updates=updates)
        self.stream, self.last_batch, self.last_digest = candidate, batch_id, digest
        self.last_response, self.total_samples = response, response['total_samples']
        return response


class InferenceService:
    def __init__(self, client, store, calibration=None):
        self.client, self.store, self.calibration = client, store, calibration

    async def start(self, handshake):
        if (handshake.get('protocol') != 'roughroute.imu.v1' or handshake.get('sample_rate_hz') != 100
                or handshake.get('acceleration_frame') != 'vqf_vehicle'):
            raise ValueError('Use roughroute.imu.v1, 100 Hz, vqf_vehicle acceleration including gravity in m/s²')
        metadata = {k: handshake.get(k) for k in ('protocol', 'client', 'sample_rate_hz', 'acceleration_frame')}
        session = InferenceSession(self, metadata)
        await asyncio.to_thread(self.store.create_session, session.id, session.metadata)
        return session

    @classmethod
    def from_env(cls):
        if os.environ.get('IMU_INFERENCE_ENABLED', '').lower() not in ('true', '1'):
            return None
        url, key = os.environ.get('BASETEN_PREDICT_URL'), os.environ.get('BASETEN_API_KEY')
        database = next((os.environ[k] for k in ('IMU_DATABASE_URL', 'DATABASE_URL', 'TIGER_DATA_URL', 'POSTGRES_URL') if os.environ.get(k)), None)
        if not url or not key or not database:
            raise ValueError('Enabled IMU inference requires BASETEN_PREDICT_URL, BASETEN_API_KEY and a database URL')
        profile = os.environ.get('IMU_CALIBRATION_PROFILE', 'none')
        calibration = None
        if profile != 'none':
            calibration = json.loads(Path(__file__).with_name('calibration_profile.json').read_text())
            if profile != calibration['id']:
                raise ValueError('Unknown IMU calibration profile')
        return cls(BasetenClient(url, key), PredictionStore(database), calibration)


def install_routes(app, service):
    from fastapi import HTTPException

    @app.get('/api/imu/status')
    def status():
        return dict(enabled=service is not None, ensemble_sha256=ENSEMBLE_SHA,
                    storage=('sqlite' if service.store.sqlite else 'postgresql') if service else None,
                    calibration_id=service.calibration['id'] if service and service.calibration else None,
                    alert_filter=DEFAULT_ALERT_FILTER)

    @app.get('/api/imu/sessions/{session_id}/predictions')
    async def predictions(session_id: str, after: int = -1, limit: int = 1000):
        if service is None:
            raise HTTPException(503, 'IMU inference is not configured')
        return await asyncio.to_thread(service.store.predictions, session_id, after, limit)

    @app.on_event('shutdown')
    async def shutdown():
        if service is not None:
            await service.client.close()
