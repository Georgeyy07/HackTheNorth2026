"""Test Camera WebSocket streaming endpoint (/ws/camera) with a real uvicorn test server."""
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


async def client_camera_session(port):
    uri = f"ws://127.0.0.1:{port}/ws/camera"
    async with websockets.connect(uri) as ws:
        # Stream 16 frames at ~16.6 fps (~60ms interval)
        for frame_num in range(1, 17):
            payload = {
                "type": "camera_frame",
                "frame_number": frame_num,
                "timestamp": int(time.time() * 1000),
                "fps": 16,
                "image": None,
            }
            await ws.send(json.dumps(payload))
            res_text = await ws.recv()
            res = json.loads(res_text)
            assert res["status"] == "ok"
            assert res["type"] == "camera_ack"
            assert res["frame"] == frame_num
            await asyncio.sleep(0.06)  # 60ms ~ 16.6 fps

        print(f"[OK] Successfully streamed 16 camera frames at 16+ fps to /ws/camera")

    # Also verify /ws/imu endpoint handles camera frame packets
    uri_imu = f"ws://127.0.0.1:{port}/ws/imu"
    async with websockets.connect(uri_imu) as ws_imu:
        payload = {
            "type": "camera_frame",
            "frame_number": 99,
            "timestamp": int(time.time() * 1000),
            "fps": 16,
        }
        await ws_imu.send(json.dumps(payload))
        res_text = await ws_imu.recv()
        res = json.loads(res_text)
        assert res["status"] == "ok"
        assert res["frame"] == 99
        print(f"[OK] Successfully received camera frame packet on /ws/imu")


def test_camera_websocket():
    app = create_app()
    test_port = 8798
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
        asyncio.run(client_camera_session(test_port))
    finally:
        server.should_exit = True
        thread.join(timeout=3)
        print("Test server stopped cleanly")


if __name__ == "__main__":
    test_camera_websocket()
    print("All Camera WebSocket end-to-end tests passed!")
