"""Optional real PostgreSQL checks in temporary tables; never modifies public rows."""
from contextlib import contextmanager
import json
import os
import uuid
import pytest
from vision_inference.store import VisionStore


def test_tiger_schema_atomic_reporting():
    url=os.environ.get('TEST_VISION_POSTGRES_URL')
    if not url:pytest.skip('Set TEST_VISION_POSTGRES_URL for temporary-table PostgreSQL verification')
    import psycopg2
    conn=psycopg2.connect(url,connect_timeout=10)
    try:
        with conn.cursor() as cur:
            cur.execute('CREATE TEMP TABLE potholes (LIKE public.potholes INCLUDING DEFAULTS)')
            cur.execute('''CREATE TEMP TABLE vision_frames (
                frame_id TEXT PRIMARY KEY,request_sha256 TEXT NOT NULL,image_sha256 TEXT NOT NULL,
                device_id TEXT NOT NULL,captured_at TIMESTAMPTZ NOT NULL,response JSONB NOT NULL)''')
            cur.execute('''CREATE TEMP TABLE vision_detections (
                id TEXT PRIMARY KEY,frame_id TEXT NOT NULL REFERENCES vision_frames(frame_id),
                class_name TEXT NOT NULL,confidence DOUBLE PRECISION NOT NULL,box JSONB NOT NULL,
                latitude DOUBLE PRECISION,longitude DOUBLE PRECISION,pothole_id TEXT)''')
            cur.execute('SET LOCAL search_path=pg_temp,public')
        class TemporaryStore(VisionStore):
            @contextmanager
            def connection(self):
                yield conn
        store=TemporaryStore(url)
        frame=dict(frame_id=str(uuid.uuid4()),device_id='test-only',frame_number=1,
                   timestamp=1700000000000,captured_at='2023-11-14T22:13:20+00:00',
                   image_sha256='test',latitude=43.,longitude=-80.)
        result=dict(detections=[dict(box=[1,2,30,40],conf=.9,cls=0,label='pothole')],provider='test',model_ref='test')
        response=store.save(frame,'test-sha',result)
        assert store.save(frame,'test-sha',result)==response
        second=dict(frame,frame_id=str(uuid.uuid4()),timestamp=1700000001000,captured_at='2023-11-14T22:13:21+00:00')
        assert store.save(second,'second-sha',result)['pothole_ids']==response['pothole_ids']
        with conn.cursor() as cur:
            cur.execute('SELECT severity,confidence,detected_by_vision,visual_box_confidence,num_visual_detections,hit_count FROM potholes')
            rows=cur.fetchall()
            assert len(rows)==1
            assert rows[0][0] is None and rows[0][1]==pytest.approx(.9) and rows[0][2] is True
            assert rows[0][3]==pytest.approx(.9) and rows[0][4:]==(2,2)
            cur.execute('SELECT COUNT(*) FROM vision_frames');assert cur.fetchone()[0]==2
            cur.execute('SELECT COUNT(*) FROM vision_detections');assert cur.fetchone()[0]==2
    finally:
        conn.rollback()
        conn.close()
