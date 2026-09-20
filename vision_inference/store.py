"""Atomic vision observations and optional reports in the existing potholes table."""
from datetime import datetime, timezone
import json
import math
import uuid
from imu_inference.store import PredictionStore


def distance_m(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371000 * 2 * math.asin(min(1, math.sqrt(h)))


def decoded(value):
    return json.loads(value) if isinstance(value, str) else value


class VisionStore(PredictionStore):
    def __init__(self, url, publish_potholes=True):
        self.publish_potholes = publish_potholes
        super().__init__(url)

    def initialize(self):
        time_type, json_type = ('TEXT', 'TEXT') if self.sqlite else ('TIMESTAMPTZ', 'JSONB')
        with self.connection() as conn:
            self.execute(conn, f'''CREATE TABLE IF NOT EXISTS vision_frames (
                frame_id TEXT PRIMARY KEY, request_sha256 TEXT NOT NULL, image_sha256 TEXT NOT NULL,
                device_id TEXT NOT NULL, captured_at {time_type} NOT NULL,
                response {json_type} NOT NULL)''')
            self.execute(conn, f'''CREATE TABLE IF NOT EXISTS vision_detections (
                id TEXT PRIMARY KEY, frame_id TEXT NOT NULL REFERENCES vision_frames(frame_id),
                class_name TEXT NOT NULL, confidence DOUBLE PRECISION NOT NULL,
                box {json_type} NOT NULL, latitude DOUBLE PRECISION, longitude DOUBLE PRECISION,
                pothole_id TEXT)''')
            self.execute(conn, 'CREATE INDEX IF NOT EXISTS vision_frames_device_time ON vision_frames(device_id, captured_at)')
            self.execute(conn, 'CREATE INDEX IF NOT EXISTS vision_detections_frame ON vision_detections(frame_id)')
            if self.publish_potholes:
                if self.sqlite:
                    self.execute(conn, '''CREATE TABLE IF NOT EXISTS potholes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, latitude REAL NOT NULL, longitude REAL NOT NULL,
                        severity TEXT DEFAULT 'UNKNOWN', timestamp TEXT DEFAULT CURRENT_TIMESTAMP)''')
                else:
                    columns = {row[0] for row in self.execute(conn, "SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='potholes'").fetchall()}
                    required = {'id','latitude','longitude','severity','confidence','detected_by_vision',
                                'visual_box_confidence','hit_count','last_hit_at','device_id','num_visual_detections','is_active'}
                    if not required <= columns:
                        raise ValueError('Vision publishing requires the existing Tiger potholes schema; use YOLO_PUBLISH_POTHOLES=false for observation-only storage')

    def lookup(self, frame_id, request_sha, conn=None):
        if conn is None:
            with self.connection() as conn:
                return self.lookup(frame_id, request_sha, conn)
        row = self.execute(conn, 'SELECT request_sha256,response FROM vision_frames WHERE frame_id=?', (frame_id,)).fetchone()
        if row:
            if row[0] != request_sha:
                raise ValueError('frame_id reused with different image or metadata')
            return decoded(row[1])
        return None

    def frame(self, frame_id):
        with self.connection() as conn:
            row = self.execute(conn, 'SELECT response FROM vision_frames WHERE frame_id=?', (frame_id,)).fetchone()
        return decoded(row[0]) if row else None

    def _publish(self, conn, frame, detections):
        if not self.publish_potholes or frame['latitude'] is None:
            return None
        potholes = [d for d in detections if d['label'] == 'pothole']
        if not potholes:
            return None
        lat, lon = frame['latitude'], frame['longitude']
        captured = frame['timestamp']
        cutoff = datetime.fromtimestamp((captured-10000)/1000, timezone.utc).isoformat()
        # Short-lived same-device report grouping. This is not object triangulation.
        nearby = self.execute(conn, '''SELECT d.pothole_id,d.latitude,d.longitude FROM vision_detections d
            JOIN vision_frames f ON f.frame_id=d.frame_id
            WHERE f.device_id=? AND f.captured_at>=? AND f.captured_at<=?
              AND d.pothole_id IS NOT NULL ORDER BY f.captured_at DESC LIMIT 200''',
            (frame['device_id'], cutoff, frame['captured_at'])).fetchall()
        confidence = max(d['conf'] for d in potholes)
        for pid, other_lat, other_lon in nearby:
            if distance_m((lat,lon),(other_lat,other_lon)) <= 8:
                if self.sqlite:
                    exists = self.execute(conn, 'SELECT id FROM potholes WHERE id=?', (pid,)).fetchone()
                else:
                    exists = self.execute(conn, '''UPDATE potholes SET hit_count=hit_count+1,
                        num_visual_detections=num_visual_detections+?, last_hit_at=?,
                        confidence=GREATEST(confidence,?), visual_box_confidence=GREATEST(visual_box_confidence,?)
                        WHERE id=? AND is_active=true RETURNING id''',
                        (len(potholes),frame['captured_at'],confidence,confidence,pid)).fetchone()
                if exists:
                    return str(pid)
        if self.sqlite:
            cursor = self.execute(conn, "INSERT INTO potholes(latitude,longitude,severity,timestamp) VALUES (?,?,'UNKNOWN',?)",
                                  (lat,lon,frame['captured_at']))
            return str(cursor.lastrowid)
        cursor = self.execute(conn, '''INSERT INTO potholes
            (latitude,longitude,severity,confidence,detected_by_vision,visual_box_confidence,
             device_id,num_visual_detections,detected_at,last_hit_at)
            VALUES (?,?,NULL,?,true,?,?,?,?,?) RETURNING id''',
            (lat,lon,confidence,confidence,frame['device_id'],len(potholes),frame['captured_at'],frame['captured_at']))
        return str(cursor.fetchone()[0])

    def save(self, frame, request_sha, result):
        with self.connection() as conn:
            # Serialize the short write transaction, including retry and nearby-report lookup.
            if self.sqlite:
                conn.execute('BEGIN IMMEDIATE')
            else:
                self.execute(conn, 'SELECT pg_advisory_xact_lock(733026)')
            cached = self.lookup(frame['frame_id'], request_sha, conn)
            if cached is not None:
                return cached
            detections = result['detections']
            pothole_id = self._publish(conn, frame, detections)
            output = dict(status='ok', type='camera_ack', frame=frame['frame_number'],
                          frame_id=frame['frame_id'], detections=detections, persisted=len(detections),
                          pothole_ids=[pothole_id] if pothole_id else [], latitude=frame['latitude'],
                          longitude=frame['longitude'], location_kind='camera_position' if frame['latitude'] is not None else None,
                          captured_at=frame['captured_at'], provider=result['provider'], model_ref=result['model_ref'],
                          checkpoint_sha256=result.get('checkpoint_sha256'),
                          computed_at=datetime.now(timezone.utc).isoformat())
            self.execute(conn, 'INSERT INTO vision_frames VALUES (?,?,?,?,?,?)',
                         (frame['frame_id'],request_sha,frame['image_sha256'],frame['device_id'],
                          frame['captured_at'],json.dumps(output,allow_nan=False)))
            for detection in detections:
                self.execute(conn, 'INSERT INTO vision_detections VALUES (?,?,?,?,?,?,?,?)',
                             (str(uuid.uuid4()),frame['frame_id'],detection['label'],detection['conf'],
                              json.dumps(detection['box']),frame['latitude'],frame['longitude'],
                              pothole_id if detection['label']=='pothole' else None))
        return output
