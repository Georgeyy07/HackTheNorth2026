"""Additive PostgreSQL/Tiger storage, with explicit SQLite for development only."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


class PredictionStore:
    def __init__(self, url):
        self.url = url
        self.sqlite = url.startswith('sqlite:///')
        if not self.sqlite and not url.startswith(('postgres://', 'postgresql://')):
            raise ValueError('Provide a PostgreSQL URL or explicit sqlite:///path')
        if self.sqlite:
            self.path = url[len('sqlite:///'):]
            if self.path == ':memory:':
                raise ValueError('Use a SQLite file so connections share committed data')
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self):
        if self.sqlite:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.execute("PRAGMA foreign_keys = ON")
        else:
            import psycopg2
            conn = psycopg2.connect(self.url, connect_timeout=5)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, conn, sql, params=()):
        cur = conn.cursor()
        cur.execute(sql if self.sqlite else sql.replace('?', '%s'), params)
        return cur

    def initialize(self):
        timestamp_type = 'TEXT' if self.sqlite else 'TIMESTAMPTZ'
        json_type = 'TEXT' if self.sqlite else 'JSONB'
        with self.connection() as conn:
            self.execute(conn, f'''CREATE TABLE IF NOT EXISTS imu_sessions (
                session_id TEXT PRIMARY KEY, created_at {timestamp_type} NOT NULL, metadata {json_type} NOT NULL)''')
            self.execute(conn, f'''CREATE TABLE IF NOT EXISTS imu_predictions (
                session_id TEXT NOT NULL REFERENCES imu_sessions(session_id),
                target_patch INTEGER NOT NULL, observed_at {timestamp_type} NOT NULL, computed_at {timestamp_type} NOT NULL,
                latitude DOUBLE PRECISION, longitude DOUBLE PRECISION,
                quality_grade INTEGER, defect_score DOUBLE PRECISION,
                disturbance INTEGER, payload {json_type} NOT NULL,
                PRIMARY KEY (session_id, target_patch))''')
            self.execute(conn, 'CREATE INDEX IF NOT EXISTS imu_predictions_observed ON imu_predictions(observed_at)')

    def create_session(self, session_id, metadata):
        with self.connection() as conn:
            self.execute(conn, 'INSERT INTO imu_sessions VALUES (?, ?, ?)',
                         (session_id, datetime.now(timezone.utc).isoformat(), json.dumps(metadata, allow_nan=False)))

    def save(self, session_id, rows):
        finals = [row for row in rows if row['is_final']]
        values = [(session_id, row['target_patch'], row['observed_at'], row['computed_at'],
                   row['latitude'], row['longitude'], row['quality_grade'], row['probability'],
                   None if row['disturbance'] is None else int(row['disturbance']),
                   json.dumps(row, allow_nan=False)) for row in finals]
        if not values:
            return 0
        with self.connection() as conn:
            if self.sqlite:
                conn.executemany("""INSERT INTO imu_predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (session_id, target_patch) DO NOTHING""", values)
            else:
                from psycopg2.extras import execute_values
                with conn.cursor() as cur:
                    execute_values(cur, """INSERT INTO imu_predictions VALUES %s
                        ON CONFLICT (session_id, target_patch) DO NOTHING""", values, page_size=1000)
        return len(finals)

    def predictions(self, session_id, after=-1, limit=1000):
        with self.connection() as conn:
            rows = self.execute(conn, '''SELECT payload FROM imu_predictions
                WHERE session_id=? AND target_patch>? ORDER BY target_patch LIMIT ?''',
                (session_id, after, min(max(limit, 1), 5000))).fetchall()
        return [json.loads(row[0]) if isinstance(row[0], str) else row[0] for row in rows]
