"""Tiger Data (PostgreSQL + TimescaleDB) database module for storing and querying road potholes.
Reads connection credentials directly from workspace .env file.
"""
import os
import sqlite3
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("tiger_db")

# Helper to automatically load workspace .env file
def load_env():
    root_dir = Path(__file__).resolve().parent.parent
    env_path = root_dir / ".env"
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()

load_env()

# Connection URL from environment
TIGER_DATA_URL = os.environ.get("DATABASE_URL") or os.environ.get("TIGER_DATA_URL") or os.environ.get("POSTGRES_URL")
USE_POSTGRES = False

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


def get_db_connection():
    """Returns a connection object to Tiger Data (PostgreSQL) or local SQLite fallback."""
    global USE_POSTGRES
    if TIGER_DATA_URL and HAS_PSYCOPG2:
        try:
            url = TIGER_DATA_URL
            conn = psycopg2.connect(url, connect_timeout=5)
            USE_POSTGRES = True
            return conn
        except Exception as e:
            logger.warning(f"Could not connect to Tiger Data PostgreSQL instance ({e}). Falling back to local SQLite database.")
    
    # Fallback to local SQLite database
    db_path = Path(__file__).resolve().parent.parent / "artifacts" / "tiger_potholes.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    USE_POSTGRES = False
    return conn


def init_db():
    """Initialize potholes table with simplified schema (id, latitude, longitude, severity, timestamp)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if USE_POSTGRES:
        try:
            # Enable TimescaleDB extension if available
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            conn.commit()
        except Exception as e:
            logger.info(f"TimescaleDB extension notice: {e}")
            conn.rollback()

        # Drop old table if columns mismatch (e.g. depth_cm exists from previous schema)
        try:
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='potholes';")
            existing_cols = [r[0] for r in cursor.fetchall()]
            if 'depth_cm' in existing_cols or 'description' in existing_cols or 'session_id' in existing_cols:
                cursor.execute("DROP TABLE IF EXISTS potholes CASCADE;")
                conn.commit()
        except Exception as e:
            conn.rollback()

        create_sql = """
        CREATE TABLE IF NOT EXISTS potholes (
            id SERIAL PRIMARY KEY,
            latitude DOUBLE PRECISION NOT NULL,
            longitude DOUBLE PRECISION NOT NULL,
            severity VARCHAR(50) DEFAULT 'MEDIUM',
            timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_sql)
        conn.commit()

        # Setup Timescale hypertable on timestamp column
        try:
            cursor.execute("SELECT create_hypertable('potholes', 'timestamp', if_not_exists => TRUE);")
            conn.commit()
        except Exception as e:
            logger.info(f"Hypertable notice: {e}")
            conn.rollback()

    else:
        # SQLite fallback schema
        try:
            cursor.execute("PRAGMA table_info(potholes);")
            cols = [r[1] for r in cursor.fetchall()]
            if 'depth_cm' in cols or 'description' in cols or 'session_id' in cols:
                cursor.execute("DROP TABLE IF EXISTS potholes;")
                conn.commit()
        except Exception:
            pass

        create_sql = """
        CREATE TABLE IF NOT EXISTS potholes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            severity TEXT DEFAULT 'MEDIUM',
            timestamp TEXT DEFAULT (datetime('now'))
        );
        """
        cursor.execute(create_sql)
        conn.commit()

    cursor.close()
    conn.close()


def get_potholes(severity: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all potholes from Tiger Data database, with optional severity filter."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT id, latitude, longitude, severity, timestamp FROM potholes WHERE 1=1"
    params = []
    
    if severity:
        query += " AND severity = ?" if not USE_POSTGRES else " AND severity = %s"
        params.append(severity)
        
    query += " ORDER BY id DESC"
    
    cursor.execute(query, params)
    
    if USE_POSTGRES:
        colnames = [desc[0] for desc in cursor.description]
        rows = [dict(zip(colnames, row)) for row in cursor.fetchall()]
    else:
        rows = [dict(row) for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    for row in rows:
        if isinstance(row.get("timestamp"), (datetime.datetime, datetime.date)):
            row["timestamp"] = row["timestamp"].isoformat()
            
    return rows


def add_pothole(
    latitude: float,
    longitude: float,
    severity: str = "MEDIUM"
) -> Dict[str, Any]:
    """Insert a new pothole record into Tiger Data database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    if USE_POSTGRES:
        sql = """
        INSERT INTO potholes (latitude, longitude, severity, timestamp)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        cursor.execute(sql, (latitude, longitude, severity, now_iso))
        new_id = cursor.fetchone()[0]
        conn.commit()
    else:
        sql = """
        INSERT INTO potholes (latitude, longitude, severity, timestamp)
        VALUES (?, ?, ?, ?);
        """
        cursor.execute(sql, (latitude, longitude, severity, now_iso))
        new_id = cursor.lastrowid
        conn.commit()
        
    cursor.close()
    conn.close()
    
    return {
        "id": new_id,
        "latitude": latitude,
        "longitude": longitude,
        "severity": severity,
        "timestamp": now_iso
    }


def update_pothole(
    pothole_id: int,
    severity: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
) -> bool:
    """Update pothole severity and/or location in Tiger Data database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    set_clauses = []
    params = []
    placeholder = "%s" if USE_POSTGRES else "?"
    
    if severity is not None:
        set_clauses.append(f"severity = {placeholder}")
        params.append(severity)
    if latitude is not None:
        set_clauses.append(f"latitude = {placeholder}")
        params.append(float(latitude))
    if longitude is not None:
        set_clauses.append(f"longitude = {placeholder}")
        params.append(float(longitude))
        
    if not set_clauses:
        conn.close()
        return False
        
    ph_clause = f"WHERE id = {placeholder}"
    params.append(pothole_id)
    sql = f"UPDATE potholes SET {', '.join(set_clauses)} {ph_clause}"
    
    cursor.execute(sql, tuple(params))
    conn.commit()
    affected = cursor.rowcount > 0
    cursor.close()
    conn.close()
    return affected


def delete_pothole(pothole_id: int) -> bool:
    """Delete a pothole record from Tiger Data database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    sql = "DELETE FROM potholes WHERE id = ?" if not USE_POSTGRES else "DELETE FROM potholes WHERE id = %s"
    cursor.execute(sql, (pothole_id,))
    conn.commit()
    affected = cursor.rowcount > 0
    cursor.close()
    conn.close()
    return affected


def seed_sample_potholes():
    """Seed sample potholes into Tiger Data database using .env credentials."""
    init_db()
    existing = get_potholes()
    if len(existing) > 0:
        logger.info(f"Tiger Data database already contains {len(existing)} potholes.")
        return
        
    sample_potholes = [
        {"latitude": 43.4723, "longitude": -80.5449, "severity": "CRITICAL"},
        {"latitude": 43.4750, "longitude": -80.5380, "severity": "HIGH"},
        {"latitude": 43.4680, "longitude": -80.5250, "severity": "MEDIUM"},
        {"latitude": 43.4785, "longitude": -80.5201, "severity": "LOW"},
        {"latitude": 39.6180, "longitude": 22.4280, "severity": "HIGH"},
        {"latitude": 39.6210, "longitude": 22.4310, "severity": "CRITICAL"}
    ]
    
    for p in sample_potholes:
        add_pothole(**p)
    logger.info("Successfully seeded potholes into Tiger Data database!")


if __name__ == "__main__":
    init_db()
    seed_sample_potholes()
    print("Potholes in Tiger Data Database:", get_potholes())
