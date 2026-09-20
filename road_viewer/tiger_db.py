"""Tiger Data (PostgreSQL + TimescaleDB) database module for storing and querying road potholes.
Reads connection credentials directly from workspace .env file.
"""
import os
import sqlite3
import datetime
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import sentry_sdk

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
                os.environ.setdefault(key.strip(), val.strip())

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

        # Create simulated_detections table (timestamp int, imu bool, yolo bool, latitude, longitude, car_id)
        create_simulated_detections_sql = """
        CREATE TABLE IF NOT EXISTS simulated_detections (
            id SERIAL,
            timestamp BIGINT NOT NULL,
            imu BOOLEAN NOT NULL DEFAULT FALSE,
            yolo BOOLEAN NOT NULL DEFAULT FALSE,
            latitude DOUBLE PRECISION NOT NULL,
            longitude DOUBLE PRECISION NOT NULL,
            car_id VARCHAR(100) NOT NULL,
            PRIMARY KEY (id, timestamp)
        );
        CREATE INDEX IF NOT EXISTS idx_simulated_detections_car_time ON simulated_detections (car_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_simulated_detections_coords ON simulated_detections (latitude, longitude);
        """
        cursor.execute(create_simulated_detections_sql)
        conn.commit()

    else:
        # Preserve existing data; schema changes require an explicit migration.
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

        # SQLite simulated_detections table
        create_simulated_detections_sqlite = """
        CREATE TABLE IF NOT EXISTS simulated_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            imu INTEGER NOT NULL DEFAULT 0,
            yolo INTEGER NOT NULL DEFAULT 0,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            car_id TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_simulated_detections_car_time ON simulated_detections (car_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_simulated_detections_coords ON simulated_detections (latitude, longitude);
        """
        cursor.execute(create_simulated_detections_sqlite)
        conn.commit()

    cursor.close()
    conn.close()


def get_potholes(severity: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all potholes from Tiger Data database, with optional severity filter."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if USE_POSTGRES:
        query = "SELECT id, latitude, longitude, severity, timestamp, confidence, detected_by_vision, detected_by_imu FROM potholes WHERE is_active = true"
    else:
        query = "SELECT id, latitude, longitude, severity, timestamp FROM potholes WHERE 1=1"
    params = []
    
    if severity:
        query += " AND severity = %s" if USE_POSTGRES else " AND severity = ?"
        params.append(severity)
        
    query += " ORDER BY id DESC"

    with sentry_sdk.start_span(op="db.query", name="tiger_db.get_potholes"):
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


def _normalize_severity_val(sev):
    if sev is None:
        return None
    try:
        return float(sev)
    except (ValueError, TypeError):
        mapping = {"CRITICAL": 9.0, "HIGH": 7.5, "MEDIUM": 5.0, "LOW": 2.5}
        return mapping.get(str(sev).upper(), 5.0)


def add_pothole(
    latitude: float,
    longitude: float,
    severity: str = "MEDIUM",
    confidence: float = 0.85,
    detected_by_vision: bool = False,
    detected_by_imu: bool = False,
    severity_score: Optional[float] = None
) -> Dict[str, Any]:
    """Insert a new pothole record into Tiger Data database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    sev_val = _normalize_severity_val(severity)
    if severity_score is None:
        severity_score = float(sev_val) if sev_val is not None else 5.0
    
    if USE_POSTGRES:
        sql = """
        INSERT INTO potholes (
            latitude, longitude, severity, timestamp, confidence, 
            detected_by_vision, detected_by_imu, severity_score, is_active
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, true)
        RETURNING id;
        """
        with sentry_sdk.start_span(op="db.query", name="tiger_db.add_pothole"):
            cursor.execute(sql, (
                latitude, longitude, sev_val, now_iso, float(confidence),
                bool(detected_by_vision), bool(detected_by_imu), float(severity_score)
            ))
        new_id = cursor.fetchone()[0]
        conn.commit()
    else:
        sql = """
        INSERT INTO potholes (latitude, longitude, severity, timestamp)
        VALUES (?, ?, ?, ?);
        """
        with sentry_sdk.start_span(op="db.query", name="tiger_db.add_pothole"):
            cursor.execute(sql, (latitude, longitude, sev_val, now_iso))
        new_id = cursor.lastrowid
        conn.commit()
        
    cursor.close()
    conn.close()
    
    return {
        "id": str(new_id),
        "latitude": latitude,
        "longitude": longitude,
        "severity": severity,
        "confidence": confidence,
        "detected_by_vision": detected_by_vision,
        "detected_by_imu": detected_by_imu,
        "severity_score": severity_score,
        "timestamp": now_iso
    }


def upsert_or_merge_pothole(
    latitude: float,
    longitude: float,
    severity: str = "MEDIUM",
    confidence: float = 0.85,
    detected_by_vision: bool = False,
    detected_by_imu: bool = False,
    radius_m: float = 15.0,
) -> Dict[str, Any]:
    """If a pothole exists within radius_m (15m), merge them and boost confidence.
    Otherwise insert a new pothole.
    """
    from alert_service.alert_math import haversine_distance_m
    active = get_potholes()

    closest_pothole = None
    min_dist = float("inf")
    for p in active:
        p_lat = float(p.get("latitude", 0))
        p_lon = float(p.get("longitude", 0))
        dist = haversine_distance_m(latitude, longitude, p_lat, p_lon)
        if dist <= radius_m and dist < min_dist:
            min_dist = dist
            closest_pothole = p

    if closest_pothole is not None:
        p_id = closest_pothole["id"]
        current_conf = float(closest_pothole.get("confidence") or 0.85)
        boosted_conf = min(0.99, round(max(current_conf, confidence) + 0.05, 2))

        sev_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        cur_sev_str = str(closest_pothole.get("severity", "MEDIUM")).upper()
        inc_sev_str = str(severity).upper()
        chosen_sev = inc_sev_str if sev_order.get(inc_sev_str, 2) > sev_order.get(cur_sev_str, 2) else cur_sev_str

        has_vision = bool(closest_pothole.get("detected_by_vision")) or detected_by_vision
        has_imu = bool(closest_pothole.get("detected_by_imu")) or detected_by_imu

        conn = get_db_connection()
        cursor = conn.cursor()
        if USE_POSTGRES:
            sql = """
            UPDATE potholes
            SET confidence = %s, severity = %s, detected_by_vision = %s, detected_by_imu = %s
            WHERE id::text = %s;
            """
            cursor.execute(sql, (boosted_conf, _normalize_severity_val(chosen_sev), has_vision, has_imu, str(p_id)))
        else:
            sql = """
            UPDATE potholes
            SET confidence = ?, severity = ?
            WHERE id = ?;
            """
            cursor.execute(sql, (boosted_conf, _normalize_severity_val(chosen_sev), p_id))
        conn.commit()
        cursor.close()
        conn.close()

        closest_pothole["confidence"] = boosted_conf
        closest_pothole["severity"] = chosen_sev
        closest_pothole["detected_by_vision"] = has_vision
        closest_pothole["detected_by_imu"] = has_imu

        return {
            "status": "merged",
            "merged_with_id": str(p_id),
            "distance_m": round(min_dist, 1),
            "confidence": boosted_conf,
            "pothole": closest_pothole,
        }
    else:
        new_record = add_pothole(
            latitude=latitude,
            longitude=longitude,
            severity=severity,
            confidence=confidence,
            detected_by_vision=detected_by_vision,
            detected_by_imu=detected_by_imu,
        )
        return {
            "status": "created",
            "pothole": new_record,
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
        params.append(_normalize_severity_val(severity))
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

    with sentry_sdk.start_span(op="db.query", name="tiger_db.update_pothole"):
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
    with sentry_sdk.start_span(op="db.query", name="tiger_db.delete_pothole"):
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


def add_simulated_detection(
    timestamp: int,
    imu: bool,
    yolo: bool,
    latitude: float,
    longitude: float,
    car_id: str,
    road_quality: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert a new detection record into simulated_detections table in Tiger Data database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    ts_val = int(timestamp)
    imu_val = bool(imu)
    yolo_val = bool(yolo)
    lat_val = float(latitude)
    lon_val = float(longitude)
    car_val = str(car_id).strip()

    if road_quality is None:
        if imu_val and yolo_val:
            road_quality_val = "bad"
        elif imu_val or yolo_val:
            road_quality_val = "medium"
        else:
            road_quality_val = "good"
    else:
        rq = str(road_quality).lower().strip()
        if rq in ("bad", "poor", "critical"):
            road_quality_val = "bad"
        elif rq in ("medium", "moderate", "fair"):
            road_quality_val = "medium"
        else:
            road_quality_val = "good"


    if USE_POSTGRES:
        sql = """
        INSERT INTO simulated_detections (timestamp, imu, yolo, latitude, longitude, car_id, road_quality)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        cursor.execute(sql, (ts_val, imu_val, yolo_val, lat_val, lon_val, car_val, road_quality_val))
        new_id = cursor.fetchone()[0]
        conn.commit()
    else:
        sql = """
        INSERT INTO simulated_detections (timestamp, imu, yolo, latitude, longitude, car_id, road_quality)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        try:
            cursor.execute(sql, (ts_val, int(imu_val), int(yolo_val), lat_val, lon_val, car_val, road_quality_val))
        except sqlite3.OperationalError:
            # Fallback if SQLite schema doesn't have road_quality column yet
            cursor.execute("""
                INSERT INTO simulated_detections (timestamp, imu, yolo, latitude, longitude, car_id)
                VALUES (?, ?, ?, ?, ?, ?);
            """, (ts_val, int(imu_val), int(yolo_val), lat_val, lon_val, car_val))
        new_id = cursor.lastrowid
        conn.commit()

    cursor.close()
    conn.close()

    return {
        "id": new_id,
        "timestamp": ts_val,
        "imu": imu_val,
        "yolo": yolo_val,
        "latitude": lat_val,
        "longitude": lon_val,
        "car_id": car_val,
        "road_quality": road_quality_val,
    }


def get_simulated_detections(
    car_id: Optional[str] = None,
    imu: Optional[bool] = None,
    yolo: Optional[bool] = None,
    order: str = "desc",
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Retrieve detections from simulated_detections table with optional filters."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT id, timestamp, imu, yolo, latitude, longitude, car_id"
    if USE_POSTGRES:
        query += ", road_quality"
    query += " FROM simulated_detections WHERE 1=1"
    
    params = []
    placeholder = "%s" if USE_POSTGRES else "?"

    if car_id is not None:
        query += f" AND car_id = {placeholder}"
        params.append(str(car_id))
    if imu is not None:
        query += f" AND imu = {placeholder}"
        params.append(bool(imu) if USE_POSTGRES else int(bool(imu)))
    if yolo is not None:
        query += f" AND yolo = {placeholder}"
        params.append(bool(yolo) if USE_POSTGRES else int(bool(yolo)))

    order_dir = "ASC" if str(order).lower() == "asc" else "DESC"
    query += f" ORDER BY timestamp {order_dir} LIMIT {int(limit)}"

    cursor.execute(query, tuple(params))

    if USE_POSTGRES:
        colnames = [desc[0] for desc in cursor.description]
        rows = [dict(zip(colnames, row)) for row in cursor.fetchall()]
    else:
        rows = [dict(row) for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    for row in rows:
        row["imu"] = bool(row.get("imu"))
        row["yolo"] = bool(row.get("yolo"))
        rq = str(row.get("road_quality", "")).lower().strip()
        if rq in ("bad", "poor", "critical"):
            row["road_quality"] = "bad"
        elif rq in ("medium", "moderate", "fair"):
            row["road_quality"] = "medium"
        elif rq == "good":
            row["road_quality"] = "good"
        else:
            if row["imu"] and row["yolo"]:
                row["road_quality"] = "bad"
            elif row["imu"] or row["yolo"]:
                row["road_quality"] = "medium"
            else:
                row["road_quality"] = "good"


    return rows


def get_simulated_detection_by_id(detection_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a single detection from simulated_detections table by its ID."""
    conn = get_db_connection()
    cursor = conn.cursor()

    placeholder = "%s" if USE_POSTGRES else "?"
    query = f"SELECT id, timestamp, imu, yolo, latitude, longitude, car_id FROM simulated_detections WHERE id = {placeholder}"

    cursor.execute(query, (int(detection_id),))

    if USE_POSTGRES:
        row = cursor.fetchone()
        if not row:
            cursor.close()
            conn.close()
            return None
        colnames = [desc[0] for desc in cursor.description]
        record = dict(zip(colnames, row))
    else:
        row = cursor.fetchone()
        if not row:
            cursor.close()
            conn.close()
            return None
        record = dict(row)

    cursor.close()
    conn.close()

    record["imu"] = bool(record.get("imu"))
    record["yolo"] = bool(record.get("yolo"))
    return record


def delete_simulated_detection(detection_id: int) -> bool:
    """Delete a record from simulated_detections table by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()

    placeholder = "%s" if USE_POSTGRES else "?"
    sql = f"DELETE FROM simulated_detections WHERE id = {placeholder}"
    cursor.execute(sql, (int(detection_id),))
    conn.commit()
    affected = cursor.rowcount > 0
    cursor.close()
    conn.close()
    return affected


def clear_simulated_detections(car_id: Optional[str] = None) -> int:
    """Clear all records or records for a specific car from simulated_detections."""
    conn = get_db_connection()
    cursor = conn.cursor()

    if car_id:
        placeholder = "%s" if USE_POSTGRES else "?"
        sql = f"DELETE FROM simulated_detections WHERE car_id = {placeholder}"
        cursor.execute(sql, (str(car_id),))
    else:
        sql = "DELETE FROM simulated_detections"
        cursor.execute(sql)

    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()
    logger.info(f"Cleared {affected} records from simulated_detections.")
    return affected


def seed_simulated_detections(clear_existing: bool = True) -> List[Dict[str, Any]]:
    """Seed the simulated_detections table with realistic multi-vehicle trajectories in Waterloo.
    
    If both imu and yolo are False, it represents normal GPS vehicle tracking.
    When imu or yolo is True, it represents a road anomaly event.
    """
    init_db()
    if clear_existing:
        clear_simulated_detections()

    base_time_ms = int(time.time() * 1000) - (120 * 1000)  # Starts 2 minutes ago
    seeded_records = []

    # Helper function to generate smooth linear interpolation between keyframes
    def interpolate_route(waypoints_def):
        full_path = []
        for i in range(len(waypoints_def) - 1):
            w1 = waypoints_def[i]
            w2 = waypoints_def[i + 1]
            steps = w1.get("steps", 10)
            for s in range(steps):
                frac = s / float(steps)
                lat = w1["lat"] + (w2["lat"] - w1["lat"]) * frac
                lon = w1["lon"] + (w2["lon"] - w1["lon"]) * frac
                imu = False
                yolo = False
                # If start waypoint has an anomaly flag, apply it at exact keyframe
                if s == 0 and (w1.get("imu") or w1.get("yolo")):
                    imu = bool(w1.get("imu", False))
                    yolo = bool(w1.get("yolo", False))
                full_path.append({"lat": round(lat, 6), "lon": round(lon, 6), "imu": imu, "yolo": yolo})
        # Add final waypoint
        last = waypoints_def[-1]
        full_path.append({
            "lat": round(last["lat"], 6),
            "lon": round(last["lon"], 6),
            "imu": bool(last.get("imu", False)),
            "yolo": bool(last.get("yolo", False)),
        })
        return full_path

    # Route 1: Car-Alpha (Fleet Patrol) - Traveling East on University Ave past University of Waterloo
    car_alpha_keys = [
        {"lat": 43.4665, "lon": -80.5510, "steps": 7},                           # Westmount & University
        {"lat": 43.4695, "lon": -80.5475, "steps": 8},                           # UW Campus Entrance
        {"lat": 43.4725, "lon": -80.5420, "imu": True, "yolo": True, "steps": 8}, # Anomaly 1: Dual confirmed pothole near Seagram Dr!
        {"lat": 43.4735, "lon": -80.5360, "steps": 7},                           # University & Hazel St
        {"lat": 43.4742, "lon": -80.5310, "imu": True, "yolo": False, "steps": 8},# Anomaly 2: Physical IMU shock (frost heave/dip)
        {"lat": 43.4752, "lon": -80.5255, "steps": 5},                           # University & King St N
    ]

    # Route 2: Car-Beta (Transit Bus #12) - Traveling West on Columbia St W
    car_beta_keys = [
        {"lat": 43.4785, "lon": -80.5220, "steps": 8},                           # Columbia & King St N
        {"lat": 43.4770, "lon": -80.5290, "imu": False, "yolo": True, "steps": 8},# Anomaly 3: YOLO vision detection (surface crack)
        {"lat": 43.4755, "lon": -80.5370, "steps": 7},                           # Columbia & Hazel St
        {"lat": 43.4740, "lon": -80.5435, "imu": True, "yolo": True, "steps": 9}, # Anomaly 4: Dual confirmed critical pothole
        {"lat": 43.4710, "lon": -80.5510, "steps": 7},                           # Columbia & Hagey Blvd
        {"lat": 43.4680, "lon": -80.5580, "steps": 5},                           # Columbia & Westmount Rd
    ]

    # Route 3: Car-Gamma (Courier Van) - Traveling North on Westmount Rd then East onto University Ave
    car_gamma_keys = [
        {"lat": 43.4570, "lon": -80.5420, "steps": 8},                           # Westmount & Erb St
        {"lat": 43.4620, "lon": -80.5450, "steps": 8},                           # Westmount mid-block
        {"lat": 43.4665, "lon": -80.5490, "steps": 7},                           # Westmount & University intersection
        {"lat": 43.4685, "lon": -80.5460, "imu": True, "yolo": True, "steps": 8}, # Anomaly 5: Dual confirmed road pothole
        {"lat": 43.4715, "lon": -80.5430, "steps": 8},                           # Near UW Engineering / Ring Rd
        {"lat": 43.4730, "lon": -80.5390, "steps": 5},                           # East campus
    ]

    fleet_routes = [
        ("Car-Alpha (Fleet Patrol)", interpolate_route(car_alpha_keys), 0),
        ("Car-Beta (Transit Bus #12)", interpolate_route(car_beta_keys), 200),
        ("Car-Gamma (Courier Van)", interpolate_route(car_gamma_keys), 400),
    ]

    conn = get_db_connection()
    cursor = conn.cursor()
    records_to_insert = []

    for car_id, path, offset_ms in fleet_routes:
        for idx, pt in enumerate(path):
            ts = base_time_ms + offset_ms + (idx * 1500)  # 1.5 seconds between pings
            imu_val = bool(pt["imu"])
            yolo_val = bool(pt["yolo"])
            if imu_val and yolo_val:
                quality = "bad"
            elif imu_val or yolo_val:
                quality = "medium"
            else:
                quality = "good"


            records_to_insert.append((ts, imu_val, yolo_val, pt["lat"], pt["lon"], car_id, quality))
            seeded_records.append({
                "timestamp": ts,
                "imu": imu_val,
                "yolo": yolo_val,
                "latitude": pt["lat"],
                "longitude": pt["lon"],
                "car_id": car_id,
                "road_quality": quality,
            })

    if USE_POSTGRES:
        sql = """
        INSERT INTO simulated_detections (timestamp, imu, yolo, latitude, longitude, car_id, road_quality)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.executemany(sql, records_to_insert)
        conn.commit()
    else:
        sql = """
        INSERT INTO simulated_detections (timestamp, imu, yolo, latitude, longitude, car_id, road_quality)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        try:
            cursor.executemany(sql, [
                (r[0], int(r[1]), int(r[2]), r[3], r[4], r[5], r[6]) for r in records_to_insert
            ])
        except sqlite3.OperationalError:
            cursor.executemany("""
                INSERT INTO simulated_detections (timestamp, imu, yolo, latitude, longitude, car_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [(r[0], int(r[1]), int(r[2]), r[3], r[4], r[5]) for r in records_to_insert])
        conn.commit()

    cursor.close()
    conn.close()

    logger.info(f"Successfully seeded {len(seeded_records)} simulated detections across 3 vehicles!")
    return seeded_records



def get_simulated_car_observations(
    car_id: Optional[str] = None,
    scenario: Optional[str] = "staggered",
    order: str = "asc",
    limit: int = 50000,
) -> List[Dict[str, Any]]:
    """Retrieve observations from simulated_car_observations table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    placeholder = "%s" if USE_POSTGRES else "?"

    clauses = []
    params = []

    if car_id:
        clauses.append(f'"carID" = {placeholder}')
        params.append(car_id)
    elif scenario and scenario != "all":
        clauses.append(f'"carID" LIKE {placeholder}')
        params.append(f"%{scenario}%")

    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    order_direction = "ASC" if order.lower() == "asc" else "DESC"
    query = f"""
        SELECT "carID", timestamp, latitude, longitude, imu_defect_detected, yolo_pothole_detected, road_quality
        FROM simulated_car_observations
        {where_sql}
        ORDER BY timestamp {order_direction}
        LIMIT {int(limit)}
    """
    cursor.execute(query, tuple(params))
    rows = []
    for r in cursor.fetchall():
        ts = r[1]
        ts_ms = int(ts.timestamp() * 1000) if hasattr(ts, 'timestamp') else ts
        ts_iso = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)

        rows.append({
            "carID": r[0],
            "car_id": r[0],
            "timestamp": ts_ms,
            "timestamp_iso": ts_iso,
            "latitude": float(r[2]),
            "longitude": float(r[3]),
            "imu_defect_detected": bool(r[4]) if r[4] is not None else False,
            "yolo_pothole_detected": bool(r[5]) if r[5] is not None else False,
            "imu": bool(r[4]) if r[4] is not None else False,
            "yolo": bool(r[5]) if r[5] is not None else False,
            "road_quality": str(r[6] or "good").lower(),
        })

    cursor.close()
    conn.close()
    return rows


def get_simulated_car_imu_samples(
    car_id: Optional[str] = None,
    timestamp: Optional[Any] = None,
    window_seconds: float = 5.0,
    stride: int = 1,
) -> Dict[str, Any]:
    """Fetch high-frequency IMU samples from simulated_car_imu_samples table around a given timestamp."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Parse target timestamp
    target_dt = None
    if timestamp is not None:
        if isinstance(timestamp, (int, float)):
            val = float(timestamp)
            target_dt = datetime.datetime.fromtimestamp(val / 1000.0 if val > 1e11 else val, tz=datetime.timezone.utc)
        elif isinstance(timestamp, str):
            try:
                val = float(timestamp)
                target_dt = datetime.datetime.fromtimestamp(val / 1000.0 if val > 1e11 else val, tz=datetime.timezone.utc)
            except ValueError:
                clean_ts = timestamp.replace("Z", "+00:00")
                target_dt = datetime.datetime.fromisoformat(clean_ts)
        elif isinstance(timestamp, datetime.datetime):
            target_dt = timestamp

    if target_dt is None:
        target_dt = datetime.datetime(2026, 9, 20, 12, 0, tzinfo=datetime.timezone.utc)

    window_s = max(0.5, min(30.0, float(window_seconds)))
    start_dt = target_dt - datetime.timedelta(seconds=window_s)
    end_dt = target_dt + datetime.timedelta(seconds=window_s)

    placeholder = "%s" if USE_POSTGRES else "?"
    clauses = [f"timestamp >= {placeholder}", f"timestamp <= {placeholder}"]
    params = [start_dt, end_dt]

    if car_id:
        clauses.append(f'("carID" = {placeholder} OR "carID" LIKE {placeholder})')
        params.extend([car_id, f"%{car_id}%"])

    where_sql = "WHERE " + " AND ".join(clauses)
    query = f"""
        SELECT 
            "carID",
            timestamp,
            source_time_s,
            accel_x,
            accel_y,
            accel_z,
            speed_mps,
            settling
        FROM simulated_car_imu_samples
        {where_sql}
        ORDER BY timestamp ASC
    """
    cursor.execute(query, tuple(params))
    raw_rows = cursor.fetchall()

    center_ms = target_dt.timestamp() * 1000.0
    samples = []
    
    step = max(1, int(stride))
    for idx in range(0, len(raw_rows), step):
        r = raw_rows[idx]
        ts_val = r[1]
        ts_ms = float(ts_val.timestamp() * 1000.0) if hasattr(ts_val, 'timestamp') else float(ts_val)
        rel_s = round((ts_ms - center_ms) / 1000.0, 3)

        ax = float(r[3]) if r[3] is not None else 0.0
        ay = float(r[4]) if r[4] is not None else 0.0
        az = float(r[5]) if r[5] is not None else 9.80665
        speed = float(r[6]) if r[6] is not None else 0.0
        settling = bool(r[7]) if r[7] is not None else False

        # Convert m/s^2 to g
        gx = round(ax / 9.80665, 3)
        gy = round(ay / 9.80665, 3)
        gz = round(az / 9.80665, 3)

        samples.append({
            "rel_s": rel_s,
            "ts_ms": ts_ms,
            "accel_x": round(ax, 3),
            "accel_y": round(ay, 3),
            "accel_z": round(az, 3),
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "speed_mps": round(speed, 2),
            "settling": settling,
        })

    cursor.close()
    conn.close()

    return {
        "car_id": car_id,
        "center_time_ms": center_ms,
        "window_seconds": window_s,
        "count": len(samples),
        "samples": samples,
    }


# Aliases for backward-compatibility
add_detection = add_simulated_detection
get_detections = get_simulated_detections
get_detection_by_id = get_simulated_detection_by_id
delete_detection = delete_simulated_detection


if __name__ == "__main__":
    init_db()
    seed_sample_potholes()
    seed_simulated_detections()
    print("Potholes count:", len(get_potholes()))
    print("Simulated Detections count:", len(get_simulated_detections(limit=500)))

