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
    SELECT "carID", min(timestamp) as min_ts, max(timestamp) as max_ts, count(*) as count
    FROM simulated_car_imu_samples
    GROUP BY "carID";
''')
for r in cur.fetchall():
    print(r)

conn.close()
