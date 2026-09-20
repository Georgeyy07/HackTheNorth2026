import urllib.request
import json

url = "http://localhost:8765/api/imu-samples?car_id=sim-waterloo-2to5-staggered-04&timestamp=2026-09-20T12:00:44.800Z&window_seconds=5"
try:
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())
        print(f"HTTP Status: {resp.status}")
        print(f"Count of samples: {data.get('count')}")
        if data.get("samples"):
            print("First sample:", data["samples"][0])
            print("Middle sample:", data["samples"][len(data["samples"])//2])
except Exception as e:
    print(f"HTTP request error: {e}")
