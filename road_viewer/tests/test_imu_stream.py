"""Test IMU WebSocket streaming endpoint (/ws/imu) with a real uvicorn test server."""
import asyncio
import io
import json
from pathlib import Path
import sys
import threading
import time

# Ensure workspace root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn
import websockets
from road_viewer.server import create_app


def run_test_server(server):
    server.run()


async def client_session(port):
    uri = f"ws://127.0.0.1:{port}/ws/imu"
    async with websockets.connect(uri) as ws:
        # 1. Send single sample
        single_payload = {
            "time": 10.5,
            "accel_x": 0.123,
            "accel_y": -0.456,
            "accel_z": 9.807,
            "gyro_x": 0.001,
            "gyro_y": 0.002,
            "gyro_z": -0.003,
            "speed": 13.4,
            "latitude": 43.4723,
            "longitude": -80.5449,
        }
        await ws.send(json.dumps(single_payload))
        res_text = await ws.recv()
        res = json.loads(res_text)
        assert res["status"] == "ok"
        assert res["received_samples"] == 1
        assert res["total_samples"] == 1
        print("[OK] Single sample handled successfully")

        # 2. Send 5x/sec 100Hz batch (20 samples for 200ms)
        batch_5hz_samples = [
            {
                "time": 11.0 + i * 0.01,
                "accel_x": round(0.05 + i * 0.001, 3),
                "accel_y": round(-0.15 + i * 0.001, 3),
                "accel_z": 9.81,
                "gyro_x": 0.005,
                "gyro_y": -0.005,
                "gyro_z": 0.0,
                "speed": 13.8,
                "latitude": 43.4723,
                "longitude": -80.5449,
            }
            for i in range(20)
        ]
        batch_5hz_payload = {
            "type": "imu_batch",
            "batch_id": 1,
            "sent_at": 1726800000000,
            "count": 20,
            "rate_hz": 5,
            "gps": {
                "latitude": 43.4723,
                "longitude": -80.5449,
                "speed": 13.8,
                "heading": 85.0,
            },
            "samples": batch_5hz_samples,
        }
        await ws.send(json.dumps(batch_5hz_payload))
        res_5hz_text = await ws.recv()
        res_5hz = json.loads(res_5hz_text)
        assert res_5hz["status"] == "ok"
        assert res_5hz["received_samples"] == 20
        assert res_5hz["total_samples"] == 21
        print("[OK] 5x/sec 100Hz batch (20 samples) handled successfully")

        # 3. Send 100Hz batch (100 samples for 1 second)
        batch_samples = [
            {
                "time": 12.0 + i * 0.01,
                "accel_x": round(0.1 + i * 0.001, 3),
                "accel_y": round(-0.2 + i * 0.001, 3),
                "accel_z": 9.81,
                "gyro_x": 0.01,
                "gyro_y": -0.01,
                "gyro_z": 0.0,
                "speed": 14.0,
                "latitude": 43.4723,
                "longitude": -80.5449,
            }
            for i in range(100)
        ]
        batch_payload = {
            "type": "imu_batch",
            "batch_id": 2,
            "sent_at": 1726800001000,
            "count": 100,
            "gps": {
                "latitude": 43.4723,
                "longitude": -80.5449,
                "speed": 14.0,
                "heading": 85.0,
            },
            "samples": batch_samples,
        }
        await ws.send(json.dumps(batch_payload))
        res_batch_text = await ws.recv()
        res_batch = json.loads(res_batch_text)
        assert res_batch["status"] == "ok"
        assert res_batch["batch_id"] == 2
        assert res_batch["received_samples"] == 100
        assert res_batch["total_samples"] == 121
        print("[OK] 100Hz batch (100 samples) handled successfully")


def test_imu_websocket():
    app = create_app()
    test_port = 8799
    config = uvicorn.Config(app, host="127.0.0.1", port=test_port, log_level="warning", ws="websockets")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=run_test_server, args=(server,), daemon=True)
    thread.start()

    # Wait for server to start
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.05)

    assert server.started, "Test server failed to start"
    print(f"Test server started on port {test_port}")

    try:
        asyncio.run(client_session(test_port))
    finally:
        server.should_exit = True
        thread.join(timeout=3)
        print("Test server stopped cleanly")


if __name__ == "__main__":
    test_imu_websocket()
    print("All IMU WebSocket end-to-end tests passed!")
