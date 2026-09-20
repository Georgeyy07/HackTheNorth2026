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
print(f"Connecting to database...")
conn = psycopg2.connect(url)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

cur.execute("SELECT count(*) FROM simulated_car_imu_samples;")
total = cur.fetchone()['count']
print(f"Total imu samples: {total}")

cur.execute('SELECT "carID", count(*) FROM simulated_car_imu_samples GROUP BY "carID";')
print("Counts per carID:")
for row in cur.fetchall():
    print(f"  {row['carID']}: {row['count']}")

cur.execute('SELECT * FROM simulated_car_imu_samples LIMIT 3;')
print("\nSample rows:")
for r in cur.fetchall():
    print(r)

conn.close()
