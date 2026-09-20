import os
import datetime
import psycopg2
import psycopg2.extras
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / ".env"
if env_path.is_file():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip()

url = os.environ.get("DATABASE_URL") or os.environ.get("TIGER_DATA_URL")
conn = psycopg2.connect(url)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

car_id = "sim-waterloo-2to5-staggered-04"
target_ts = "2026-09-20T12:00:44.800Z"
window_s = 5.0

dt = datetime.datetime.fromisoformat(target_ts.replace('Z', '+00:00'))

start_dt = dt - datetime.timedelta(seconds=window_s)
end_dt = dt + datetime.timedelta(seconds=window_s)

cur.execute("""
    SELECT 
        EXTRACT(EPOCH FROM timestamp) * 1000 AS ts_ms,
        source_time_s,
        accel_x,
        accel_y,
        accel_z,
        speed_mps
    FROM simulated_car_imu_samples
    WHERE ("carID" = %s OR "carID" LIKE %s)
      AND timestamp >= %s
      AND timestamp <= %s
    ORDER BY timestamp ASC;
""", (car_id, f"%{car_id}%", start_dt, end_dt))

rows = cur.fetchall()
print(f"Query returned {len(rows)} samples.")
if rows:
    center_ms = dt.timestamp() * 1000
    print(f"Center ts_ms: {center_ms}")
    print("First sample relative time:", (rows[0]['ts_ms'] - center_ms) / 1000.0)
    print("Last sample relative time:", (rows[-1]['ts_ms'] - center_ms) / 1000.0)

conn.close()
