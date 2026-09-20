"""Verify FastAPI -> configured YOLO service -> database with an actual image.

Creates one vision frame observation. No GPS is supplied, so it cannot create
potholes on the live map. HTTP/WebSocket retries must return the same record.
"""
import argparse
import base64
import json
from pathlib import Path
import sys
import time
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    args=p.parse_args()
    from fastapi.testclient import TestClient
    from road_viewer.server import create_app
    frame_id=str(uuid.uuid4())
    payload=dict(type='camera_frame',frame_id=frame_id,device_id='vision-smoke',frame_number=1,
                 timestamp=int(time.time()*1000),image=base64.b64encode(args.image.read_bytes()).decode())
    with TestClient(create_app()) as client:
        status=client.get('/api/vision/status').json()
        if not status['enabled']:raise RuntimeError('Enable and configure YOLO inference first')
        result=client.post('/api/vision/predict',json=payload)
        result.raise_for_status();result=result.json()
        assert not result['pothole_ids']
        assert client.get('/api/vision/frames/'+frame_id).json()==result
        with client.websocket_connect('/ws/camera') as ws:
            ws.send_json(payload)
            assert ws.receive_json()==result
        print(json.dumps(dict(frame_id=frame_id,storage=status['storage'],provider=result['provider'],
                              checkpoint_sha256=result['checkpoint_sha256'],detections=len(result['detections']),
                              retry_idempotent=True,potholes_created=0,
                              imu_enabled=client.get('/api/imu/status').json()['enabled']),indent=2))


if __name__=='__main__':main()
