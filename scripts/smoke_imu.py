"""Send prepared recording samples through FastAPI -> Baseten -> configured database.

This creates a new inference session and writes real predictions to the configured
IMU database. Use explicit SQLite for a development smoke test.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True, help='Prepared samples.parquet with accel_x/y/z and speed')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=320)
    args = parser.parse_args()
    if args.start < 0 or not 48 <= args.count <= 1024:
        parser.error('start must be nonnegative and count must be 48..1024')
    import pandas as pd
    from fastapi.testclient import TestClient
    from road_viewer.server import create_app
    origin_ms = int(time.time()*1000)
    frame = pd.read_parquet(args.samples).iloc[args.start:args.start+args.count]
    if len(frame) != args.count:
        parser.error('Recording is shorter than the requested slice')
    samples = [dict(time=i/100, available_at_ms=origin_ms+i*10,
                    **{key: float(row[key]) if pd.notna(row[key]) else None
                       for key in ('accel_x', 'accel_y', 'accel_z', 'speed')})
               for i, (_, row) in enumerate(frame.iterrows())]
    with TestClient(create_app()) as client:
        status = client.get('/api/imu/status').json()
        if not status['enabled']:
            raise RuntimeError('Configure IMU_INFERENCE_ENABLED, Baseten credentials and a database URL first')
        with client.websocket_connect('/ws/imu') as ws:
            ws.send_json(dict(type='handshake', client='prepared-recording-smoke', protocol='roughroute.imu.v1',
                              acceleration_frame='vqf_vehicle', sample_rate_hz=100))
            ready = ws.receive_json()
            if ready.get('status') != 'ready':
                raise RuntimeError(ready)
            batch = dict(type='imu_batch', batch_id=1, samples=samples)
            start = time.monotonic();ws.send_json(batch);result = ws.receive_json()
            elapsed = time.monotonic()-start
            if result.get('status') != 'ok':
                raise RuntimeError(result)
            ws.send_json(batch)
            assert ws.receive_json() == result, 'Duplicate retry changed the response'
            saved = client.get(f'/api/imu/sessions/{ready["session_id"]}/predictions').json()
            assert len(saved) == args.count//16-2 == result['persisted']
            assert all(r['ensemble_sha256'] == status['ensemble_sha256'] for r in saved)
            report = dict(session_id=ready['session_id'], storage=status['storage'], samples=args.count,
                          finalized_predictions=len(saved), calibration_id=status['calibration_id'],
                          request_seconds=elapsed, retry_idempotent=True)
            print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
