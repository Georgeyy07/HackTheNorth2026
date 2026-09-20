"""Atomically import reviewed fleet CSVs into Tiger. Requires DATABASE_URL."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from collections import Counter
import psycopg2

COLUMNS=['carID','timestamp','latitude','longitude','imu_defect_detected','yolo_pothole_detected','road_quality']
TABLE='public.simulated_car_observations'
SCHEMA='''CREATE TABLE IF NOT EXISTS public.simulated_car_observations (
    "carID" TEXT NOT NULL,
    "timestamp" TIMESTAMPTZ NOT NULL,
    latitude DOUBLE PRECISION NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude DOUBLE PRECISION NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    imu_defect_detected BOOLEAN,
    yolo_pothole_detected BOOLEAN,
    road_quality TEXT NOT NULL CHECK (road_quality IN ('good','medium','bad')),
    PRIMARY KEY ("carID","timestamp")
);
CREATE INDEX IF NOT EXISTS simulated_car_observations_timestamp_idx
ON public.simulated_car_observations ("timestamp");'''


def inspect_bundle(root):
    manifest=json.loads((root/'manifest.json').read_text())
    expected={};qualities=Counter();ids=[]
    for variant in ('staggered','cascade'):
        path=root/f'{variant}.csv'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['variants'][variant]['csv_sha256']:
            raise ValueError(f'CSV hash mismatch: {variant}')
        with path.open(newline='') as f:
            reader=csv.DictReader(f)
            if reader.fieldnames!=COLUMNS:raise ValueError('Unexpected CSV columns')
            rows=list(reader)
        plans=manifest['variants'][variant]['cars'];allowed={c['carID'] for c in plans}
        if any(not c.startswith('sim-') for c in allowed):raise ValueError('Expected synthetic car IDs')
        if len(rows)!=manifest['variants'][variant]['rows']:raise ValueError('Row count mismatch')
        for row in rows:
            if row['road_quality'] not in ('good','medium','bad'):
                raise ValueError('road_quality must be a lowercase good/medium/bad string, never NULL or numeric')
            if row['carID'] not in allowed:raise ValueError('Unexpected car ID')
            if any(row[k] not in ('true','false','') for k in ('imu_defect_detected','yolo_pothole_detected')):
                raise ValueError('Invalid boolean value')
            qualities[row['road_quality']]+=1
        actual=Counter(row['carID'] for row in rows)
        if actual!={c['carID']:c['rows'] for c in plans}:raise ValueError('Per-car counts differ from manifest')
        expected[variant]=len(rows);ids.extend(allowed)
    if len(ids)!=len(set(ids)):raise ValueError('Car IDs overlap between variants')
    return expected,qualities,sorted(ids)


def run(root, replace_bundle=None):
    counts,qualities,ids=inspect_bundle(root)
    old_ids=inspect_bundle(replace_bundle)[2] if replace_bundle else []
    if set(ids)&set(old_ids):raise ValueError('Replacement must use new car IDs')
    removed=0
    url=os.environ.get('DATABASE_URL') or os.environ.get('TIGER_DATA_URL')
    if not url:raise ValueError('Set DATABASE_URL or TIGER_DATA_URL privately')
    conn=psycopg2.connect(url,connect_timeout=10)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL lock_timeout = '10s'")
                cur.execute("SET LOCAL statement_timeout = '120s'")
                cur.execute(SCHEMA)
                cur.execute("SELECT column_name,data_type FROM information_schema.columns WHERE table_schema='public' AND table_name='simulated_car_observations' ORDER BY ordinal_position")
                expected_types=['text','timestamp with time zone','double precision','double precision','boolean','boolean','text']
                if cur.fetchall()!=list(zip(COLUMNS,expected_types)):raise ValueError('Existing destination has incompatible columns')
                cur.execute(f'CREATE TEMP TABLE fleet_import (LIKE {TABLE} INCLUDING ALL) ON COMMIT DROP')
                for variant in counts:
                    with (root/f'{variant}.csv').open() as f:
                        cur.copy_expert("COPY fleet_import FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')",f)
                cur.execute(f'INSERT INTO {TABLE} SELECT * FROM fleet_import ON CONFLICT ("carID","timestamp") DO NOTHING')
                inserted=cur.rowcount
                # A retry may skip identical rows; conflicting values must roll back.
                cur.execute(f'''SELECT count(*) FROM fleet_import s JOIN {TABLE} d
                    USING ("carID","timestamp") WHERE
                    ROW(s.latitude,s.longitude,s.imu_defect_detected,s.yolo_pothole_detected,s.road_quality)
                    IS DISTINCT FROM ROW(d.latitude,d.longitude,d.imu_defect_detected,d.yolo_pothole_detected,d.road_quality)''')
                if cur.fetchone()[0]:raise ValueError('Existing rows differ from this bundle; use a new run ID')
                cur.execute(f'SELECT "carID",count(*) FROM {TABLE} WHERE "carID"=ANY(%s) GROUP BY "carID"',(ids,))
                saved=dict(cur.fetchall())
                cur.execute('SELECT "carID",count(*) FROM fleet_import GROUP BY "carID"')
                if saved!=dict(cur.fetchall()):raise ValueError('Stored row counts differ from bundle')
                if replace_bundle:
                    # Remove only exact keys from the explicitly replaced bundle.
                    cur.execute(f'CREATE TEMP TABLE fleet_old (LIKE {TABLE} INCLUDING ALL) ON COMMIT DROP')
                    for variant in ('staggered','cascade'):
                        with (replace_bundle/f'{variant}.csv').open() as f:
                            cur.copy_expert("COPY fleet_old FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')",f)
                    cur.execute(f'''SELECT count(*) FROM {TABLE} d WHERE d."carID"=ANY(%s)
                        AND NOT EXISTS (SELECT 1 FROM fleet_old s WHERE s."carID"=d."carID" AND s."timestamp"=d."timestamp")''',(old_ids,))
                    if cur.fetchone()[0]:raise ValueError('Unexpected extra rows under old car IDs; refusing partial replacement')
                    cur.execute(f'DELETE FROM {TABLE} d USING fleet_old s WHERE d."carID"=s."carID" AND d."timestamp"=s."timestamp"')
                    removed=cur.rowcount
        # Verify committed data through a separate connection.
        with psycopg2.connect(url,connect_timeout=10) as verify:
            with verify.cursor() as cur:
                cur.execute(f'SELECT road_quality,count(*) FROM {TABLE} WHERE "carID"=ANY(%s) GROUP BY road_quality',(ids,))
                actual=dict(cur.fetchall())
                if actual!=dict(qualities):raise RuntimeError('Committed quality counts differ')
                cur.execute(f'''SELECT count(*),count(*) FILTER (WHERE imu_defect_detected IS TRUE),
                    count(*) FILTER (WHERE yolo_pothole_detected IS TRUE),count(*) FILTER (WHERE yolo_pothole_detected IS NULL)
                    FROM {TABLE} WHERE "carID"=ANY(%s)''',(ids,))
                total,imu,yolo,unknown=cur.fetchone()
                assert total==sum(counts.values())
                if old_ids:
                    cur.execute(f'SELECT count(*) FROM {TABLE} WHERE "carID"=ANY(%s)',(old_ids,))
                    if cur.fetchone()[0]:raise RuntimeError('Old scenario rows remain')
        report=dict(imported_at=datetime.now(timezone.utc).isoformat(),table=TABLE,variants=counts,
                    cars=len(ids),rows=total,newly_inserted=inserted,already_present=total-inserted,
                    road_quality_type='text',road_quality_counts=actual,imu_positive_rows=imu,
                    yolo_positive_rows=yolo,yolo_unknown_rows=unknown,committed_verified=True,
                    car_ids=ids,bundle=str(root.resolve()),replaced_bundle=str(replace_bundle) if replace_bundle else None,
                    replaced_car_ids=old_ids,removed_old_rows=removed)
        (root/'tiger_import_receipt.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally:conn.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('bundle',type=Path)
    parser.add_argument('--replace-bundle',type=Path,help='Atomically remove the exact previous scenario rows')
    args=parser.parse_args();run(args.bundle,args.replace_bundle)
