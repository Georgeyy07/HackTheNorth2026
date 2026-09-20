import asyncio
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from imu_inference.client import ENSEMBLE_SHA, validate_prediction
from imu_inference.calibration import calibrated_update
from imu_inference.service import InferenceService
from imu_inference.store import PredictionStore
from imu_inference.stream import RoadStream

HANDSHAKE = dict(protocol='roughroute.imu.v1', sample_rate_hz=100, acceleration_frame='vqf_vehicle')
CAL = json.loads((Path(__file__).resolve().parents[1]/'calibration_profile.json').read_text())


def samples(n, start=0):
    return [dict(time=i/100, available_at_ms=1700000000000+i*10, accel_x=.1,
                 accel_y=.2, accel_z=9.81, speed=0, segment=0, latitude=43.1, longitude=-80.5,
                 gps_timestamp_ms=1700000000000, gps_received_at_ms=1700000000000)
            for i in range(start, start+n)]


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.fail = False
    async def predict(self, payload):
        self.calls += 1
        if self.fail:
            raise RuntimeError('cloud failed')
        n = len(payload['windows'])
        valid = np.array(payload['masks'])[..., :3].all(-1).reshape(n, 64, 16).all(-1)
        return dict(ensemble_sha256=ENSEMBLE_SHA, patch_indices=[61, 62, 63],
                    quality_probability=np.tile([.1,.2,.7], (n,3,1)).tolist(),
                    disturbance_probability=np.tile([.9,.6,.3], (n,1)).tolist(),
                    patch_valid=valid[:, -3:].tolist())
    async def close(self):
        pass


def service(tmp_path, calibration=CAL):
    return InferenceService(FakeClient(), PredictionStore('sqlite:///'+str(tmp_path/'imu.db')), calibration)


def test_consensus_calibration_and_final_only_kalman():
    async def run():
        stream = RoadStream(CAL)
        rows = await stream.push(samples(64), FakeClient())
        finals = [r for r in rows if r['is_final']]
        assert [r['target_patch'] for r in finals] == [0,1]
        # Chronological votes .3, .6, .9 weighted 1,2,3; same as original code.
        assert finals[0]['original_probability'] == pytest.approx(.7)
        assert finals[0]['probability'] == pytest.approx(.7)
        expected = calibrated_update(dict(valid=True, quality_probability=[.1,.2,.7]), CAL)
        assert finals[0]['quality_probability'] == pytest.approx(expected['quality_probability'])
        assert finals[0]['original_quality_probability'] == pytest.approx([.1,.2,.7])
        assert stream.alerts.last_target == 1
        assert finals[0]['latitude'] == 43.1
        assert finals[0]['observed_at'] == '2023-11-14T22:13:20+00:00'
        assert stream.last_index == 63  # time=0 and speed=0 are valid
    asyncio.run(run())


def test_chunking_equivalence_and_bounded_state():
    async def run():
        full, chunk = RoadStream(CAL), RoadStream(CAL)
        a = await full.push(samples(2000), FakeClient())
        b = []
        for i in range(0, 2000, 20):
            b += await chunk.push(samples(20, i), FakeClient())
        for row in a+b:
            row.pop('computed_at')
        assert a == b
        assert len(chunk.votes) == 2 and len(chunk.locations) <= 3
        assert chunk.window.shape == (1024,4)
    asyncio.run(run())


def test_gap_resets_filter_and_censors_event():
    async def run():
        stream = RoadStream()
        await stream.push(samples(64), FakeClient())
        stream.alerts.active = True
        stream.alerts.event_start = 0
        rows = await stream.push(samples(64, 96), FakeClient())
        missing = [r for r in rows if r['is_final'] and r['target_patch'] in (4,5)]
        assert len(missing) == 2
        assert all(not r['valid'] and r['probability'] is None for r in missing)
        assert missing[0]['event_transition'] == 'censored'
    asyncio.run(run())


def test_db_cloud_failures_retry_and_session_isolation(tmp_path):
    async def run():
        svc = service(tmp_path)
        session = await svc.start(HANDSHAKE)
        batch = dict(batch_id=1, samples=samples(64))
        svc.client.fail = True
        with pytest.raises(RuntimeError):
            await session.process(batch)
        assert session.stream.last_index == -1
        svc.client.fail = False
        save = svc.store.save
        def broken(*args):
            raise RuntimeError('database failed')
        svc.store.save = broken
        with pytest.raises(RuntimeError):
            await session.process(batch)
        assert session.stream.last_index == -1
        svc.store.save = save
        response = await session.process(batch)
        calls = svc.client.calls
        assert await session.process(batch) == response
        assert svc.client.calls == calls
        assert len(svc.store.predictions(session.id)) == 2
        changed = copy.deepcopy(batch); changed['samples'][0]['speed'] = 10
        with pytest.raises(ValueError, match='reused'):
            await session.process(changed)
        other = await svc.start(HANDSHAKE)
        assert other.stream.alerts.last_target == -1 and other.id != session.id
        assert not svc.store.predictions(other.id)
    asyncio.run(run())


def test_atomic_store_and_idempotency(tmp_path):
    async def run():
        svc = service(tmp_path)
        session = await svc.start(HANDSHAKE)
        response = await session.process(dict(batch_id=1, samples=samples(64)))
        rows = [r for r in response['updates'] if r['is_final']]
        svc.store.save(session.id, rows)
        assert len(svc.store.predictions(session.id)) == 2
        broken = copy.deepcopy(rows);broken[0]['target_patch'] = 100;broken[1].pop('computed_at')
        with pytest.raises(KeyError):
            svc.store.save(session.id, broken)
        assert len(svc.store.predictions(session.id)) == 2
    asyncio.run(run())


@pytest.mark.parametrize('change', [dict(time=.003), dict(speed=-1), dict(accel_z=float('inf')),dict(available_at_ms=0)])
def test_invalid_batch_no_partial_progress(tmp_path, change):
    async def run():
        svc = service(tmp_path); session = await svc.start(HANDSHAKE)
        data = samples(64);data[-1].update(change)
        with pytest.raises(ValueError):
            await session.process(dict(batch_id=1, samples=data))
        assert session.stream.last_index == -1 and svc.client.calls == 0
    asyncio.run(run())


def test_gps_future_stale_missing_and_calibration_none():
    async def run():
        data = samples(64)
        for s in data:
            s['gps_timestamp_ms'] += 100000
        rows = await RoadStream().push(data, FakeClient())
        assert all(r['latitude'] is None and r['calibration_id'] is None for r in rows)
        assert rows[-1]['quality_probability'] == pytest.approx([.1,.2,.7])
        data = samples(64)
        for s in data:
            s['gps_timestamp_ms'] -= 4000
        assert all(r['latitude'] is None for r in await RoadStream().push(data, FakeClient()))
    asyncio.run(run())


def test_no_postgres_fallback(monkeypatch, tmp_path):
    import psycopg2
    def fail(*args, **kwargs):
        raise psycopg2.OperationalError('unavailable')
    monkeypatch.setattr(psycopg2, 'connect', fail)
    with pytest.raises(psycopg2.OperationalError):
        PredictionStore('postgresql://unavailable/test')


def test_reject_unexpected_model():
    with pytest.raises(ValueError, match='Unexpected'):
        validate_prediction({'ensemble_sha256':'wrong'}, 1)


def test_fastapi_websocket_persists(tmp_path):
    from fastapi.testclient import TestClient
    from road_viewer.server import create_app
    svc = service(tmp_path)
    with TestClient(create_app(inference_service=svc)) as client:
        assert client.get('/api/imu/status').json()['storage'] == 'sqlite'
        with client.websocket_connect('/ws/imu') as ws:
            ws.send_json(dict(type='handshake', **HANDSHAKE))
            ready = ws.receive_json();assert ready['inference']
            ws.send_json(dict(type='imu_batch', batch_id=1, samples=samples(64)))
            result = ws.receive_json();assert result['persisted'] == 2
            saved = client.get('/api/imu/sessions/'+ready['session_id']+'/predictions').json()
            assert len(saved) == 2 and saved[0]['calibration_id'] == CAL['id']
            ws.send_json(dict(type='imu_batch', batch_id=1, samples=samples(64)))
            assert ws.receive_json() == result


def test_legacy_table_is_not_dropped(tmp_path, monkeypatch):
    import sqlite3
    from road_viewer import tiger_db
    path = tmp_path/'legacy.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE potholes (id INTEGER, latitude REAL, longitude REAL, severity TEXT, timestamp TEXT, description TEXT)')
        conn.execute("INSERT INTO potholes VALUES (1, 43, -80, 'HIGH', '2026-01-01', 'preserve me')")
    monkeypatch.setattr(tiger_db, 'USE_POSTGRES', False)
    monkeypatch.setattr(tiger_db, 'get_db_connection', lambda: sqlite3.connect(path))
    tiger_db.init_db()
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT description FROM potholes').fetchone()[0] == 'preserve me'
