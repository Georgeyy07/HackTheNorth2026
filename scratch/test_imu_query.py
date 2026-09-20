import os
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

cur.execute('''
    SELECT "carID", timestamp, latitude, longitude, imu_defect_detected, yolo_pothole_detected
    FROM simulated_car_observations
    WHERE imu_defect_detected = true AND yolo_pothole_detected = true
    LIMIT 1;
''')
obs = cur.fetchone()
print(f"Anomaly observation: {obs}")

car_id = obs['carID']
ts = obs['timestamp']

# Query +- 5 seconds of imu samples around ts!
cur.execute('''
    SELECT timestamp, source_time_s, accel_x, accel_y, accel_z, speed_mps
    FROM simulated_car_imu_samples
    WHERE "carID" = %s
      AND timestamp >= %s - interval '5 seconds'
      AND timestamp <= %s + interval '5 seconds'
    ORDER BY timestamp ASC;
''', (car_id, ts, ts))
rows = cur.fetchall()
print(f"Found {len(rows)} IMU samples in +-5s window around anomaly!")
if rows:
    z_vals = [r['accel_z'] for r in rows]
    print(f"min Z: {min(z_vals):.2f}, max Z: {max(z_vals):.2f}, avg Z: {sum(z_vals)/len(z_vals):.2f}")
    print("First 3 samples:")
    for r in rows[:3]:
        print(f"  {r['timestamp']} | X:{r['accel_x']:.2f} Y:{r['accel_y']:.2f} Z:{r['accel_z']:.2f}")

conn.close()
