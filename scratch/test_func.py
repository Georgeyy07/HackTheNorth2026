import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from road_viewer.tiger_db import get_simulated_car_imu_samples

res = get_simulated_car_imu_samples(
    car_id="sim-waterloo-2to5-staggered-04",
    timestamp="2026-09-20T12:00:44.800Z",
    window_seconds=5.0
)
print("Function test success!")
print(f"Count: {res['count']}")
if res['samples']:
    print("Sample 0:", res['samples'][0])
    print("Sample middle:", res['samples'][len(res['samples'])//2])
    print("Sample last:", res['samples'][-1])
