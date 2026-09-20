"""Generate synthetic multi-car replays from one recorded route; no database writes.

python scripts/generate_fleet_scenarios.py --help
"""
import argparse
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

COLUMNS = ['carID', 'timestamp', 'latitude', 'longitude', 'imu_defect_detected',
           'yolo_pothole_detected', 'road_quality']
SCHEMA = '''-- Synthetic copies of ONE drive, never independent corroborating vehicles.
CREATE TABLE IF NOT EXISTS simulated_car_observations (
    "carID" TEXT NOT NULL,
    "timestamp" TIMESTAMPTZ NOT NULL,
    latitude DOUBLE PRECISION NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude DOUBLE PRECISION NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    imu_defect_detected BOOLEAN,
    yolo_pothole_detected BOOLEAN,
    road_quality TEXT CHECK (road_quality IN ('good', 'medium', 'bad')),
    PRIMARY KEY ("carID", "timestamp")
);
CREATE INDEX IF NOT EXISTS simulated_car_observations_timestamp_idx
ON simulated_car_observations ("timestamp");
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stationary_intervals(samples, threshold=.3, minimum_s=2.):
    """Missing speed or missing timesteps break a stop; never infer zero speed."""
    times = samples.time_s.to_numpy(float)
    speeds = samples.speed.to_numpy(float)
    step = .01
    result = []; start = None
    for i, (t, v) in enumerate(zip(times, speeds)):
        valid = np.isfinite(v) and 0 <= v <= threshold
        continuous = i == 0 or 0 < t-times[i-1] <= step*1.5
        if start is not None and (not valid or not continuous):
            end = times[i-1]+step
            if end-start >= minimum_s-1e-8: result.append((float(start), float(end)))
            start = None
        if valid and start is None: start = t
    if start is not None and times[-1]+step-start >= minimum_s-1e-8:
        result.append((float(start), float(times[-1]+step)))
    return result


def yolo_for_patch(vision, start_s, end_s):
    """Any positive frame in the target patch; NULL outside observed coverage."""
    if vision is None: return None
    times, positives, cadence = vision
    a = bisect_left(times, start_s); b = bisect_left(times, end_s)
    if a == b: return None
    if any(positives[a:b]): return True
    # A negative needs coverage across the patch, not just an isolated frame.
    selected = times[a:b]
    if selected[0]-start_s > cadence*1.6 or end_s-selected[-1] > cadence*1.6:
        return None
    if len(selected)>1 and np.max(np.diff(selected)) > cadence*1.6: return None
    return False


def load_vision(path):
    if not path.is_file(): return None, {'available': False, 'reason': 'No video inference export'}
    raw = path.read_bytes()
    data = json.loads(raw); offset = data.get('video_offset_s')
    if offset is None or not np.isfinite(offset):
        return None, {'available': False, 'reason': 'Unknown video/IMU alignment'}
    frames = data['frames']; fps = float(data['fps'])
    if not frames or fps<=0: raise ValueError(f'Invalid vision export: {path}')
    times = [float(f['video_time_s'])+offset for f in frames]
    if any(b<=a for a,b in zip(times,times[1:])): raise ValueError('Video timestamps must increase')
    positives = [any(d.get('label')=='pothole' for d in f['detections']) for f in frames]
    return (times, positives, 1/fps), dict(available=True, source=str(path), sha256=hashlib.sha256(raw).hexdigest(),
        alignment=data.get('alignment'), video_offset_s=offset, frames=len(frames),
        checkpoint_sha256=data.get('checkpoint_sha256'))


def select_start(rows, stops, samples, minimum_s):
    """Start with a mappable patch entirely at rest and >= minimum_s rest ahead."""
    times = samples.time_s.to_numpy(float); speed = samples.speed.to_numpy(float)
    for a,b in stops:
        for row in rows:
            t = row['_source_end_s']
            if row['_source_start_s'] < a-1e-8 or t+minimum_s > b+1e-8: continue
            sample_index = np.searchsorted(times,t-.01+1e-7,side='right')-1
            return row, dict(stop_start_s=a, stop_end_s=b, source_start_s=t,
                             stationary_remaining_s=b-t, speed_mps=float(speed[sample_index]))
    raise ValueError('No mappable stationary start with sufficient remaining stop time')


def load_route(imu_root, vision_root, sessions, threshold, minimum_s):
    manifest = json.loads((imu_root/'manifest.json').read_text())
    metadata = {s['session_id']:s for s in manifest['sessions']}
    vision_manifest = vision_root/'manifest.json'
    vision_sessions = ({s['session_id']:s for s in json.loads(vision_manifest.read_text())['sessions']}
                       if vision_manifest.is_file() else {})
    route = []; starts = []; sources = []; boundaries = []; previous_end = None
    for name in sessions:
        meta = metadata[name]; folder = imu_root/name
        origin_us = int(meta['timestamp_origin_unix_ns'])//1000
        if previous_end is not None:
            gap_s=(origin_us-previous_end)/1e6
            if gap_s < -.01: raise ValueError(f'Sessions overlap or are out of order: {name}')
            boundaries.append(dict(next_session=name,recording_gap_s=gap_s))
        previous_end=origin_us+round(meta['duration_s']*1e6)
        samples_path=folder/'samples.parquet';predictions_path=folder/'final_predictions.jsonl'
        for path in (samples_path,predictions_path):
            expected_hash=meta.get('outputs_sha256',{}).get(path.name)
            if expected_hash and sha(path)!=expected_hash:raise ValueError(f'Input hash mismatch: {path}')
        samples=pd.read_parquet(samples_path)
        if len(samples)!=meta['samples']:raise ValueError(f'Sample count mismatch: {name}')
        times=samples.time_s.to_numpy(float)
        if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Invalid sample clock')
        stops=stationary_intervals(samples,threshold,minimum_s)
        vision,vision_meta=load_vision(vision_root/name/'vision.json')
        if vision is not None and name in vision_sessions:
            camera_meta=vision_sessions[name]
            for key in ('source_jsonl_sha256','timestamp_origin_unix_ns'):
                if camera_meta.get(key)!=meta.get(key):raise ValueError(f'Camera and IMU source mismatch for {name}: {key}')
        predictions=[json.loads(line) for line in predictions_path.read_text().splitlines() if line]
        expected=max(0,len(samples)//16-2)
        if len(predictions)!=expected or [r['target_patch'] for r in predictions]!=list(range(expected)):
            raise ValueError(f'Incomplete finalized patch sequence: {name}')
        rows=[];no_gps=0
        for r in predictions:
            if not r['is_final']:raise ValueError('Provisional results must not be exported as final')
            lat,lon=r.get('latitude'),r.get('longitude')
            if not r.get('target_gps_valid') or lat is None or lon is None:
                no_gps+=1;continue
            if not np.isfinite([lat,lon]).all() or not (-90<=lat<=90 and -180<=lon<=180):raise ValueError('Invalid GPS')
            valid=r['valid'];quality=r.get('quality_name') if valid else None
            if quality not in (None,'good','medium','bad'):raise ValueError('Expected three quality categories')
            defect=r.get('disturbance') if valid else None
            if defect is not None and type(defect) is not bool:raise ValueError('Expected boolean defect decision')
            rows.append(dict(latitude=lat,longitude=lon,imu_defect_detected=defect,
                yolo_pothole_detected=yolo_for_patch(vision,r['start_s'],r['end_s']),road_quality=quality,
                _source_us=origin_us+round(r['end_s']*1e6),_session=name,
                _source_start_s=r['start_s'],_source_end_s=r['end_s']))
        first,proof=select_start(rows,stops,samples,minimum_s)
        starts.append(dict(session=name,source_us=first['_source_us'],**proof))
        route.extend(rows)
        sources.append(dict(session=name,origin_unix_us=origin_us,duration_s=meta['duration_s'],
            stationary_intervals=stops,finalized_patches=len(predictions),mappable_rows=len(rows),dropped_without_fresh_gps=no_gps,
            samples_sha256=sha(samples_path),predictions_sha256=sha(predictions_path),
            ensemble_sha256=meta.get('ensemble_sha256'),calibration=meta.get('calibration'),
            source_jsonl_sha256=meta.get('source_jsonl_sha256'),vision=vision_meta))
    if any(b['_source_us']<=a['_source_us'] for a,b in zip(route,route[1:])):raise ValueError('Non-increasing route timeline')
    return route,starts,sources,boundaries


def utc_string(us):
    return (datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(microseconds=us)).isoformat(timespec='microseconds').replace('+00:00','Z')


def materialize(route,starts,variant,run_id,start_us,cars,headway_s):
    plans=[];rows=[]
    selections=starts if variant=='staggered' else [starts[0]]*cars
    for i,start in enumerate(selections):
        car=f'sim-{run_id}-{variant}-{i+1:02d}'
        delay_us=round(i*headway_s*1e6) if variant=='cascade' else 0
        chosen=[r for r in route if r['_source_us']>=start['source_us']]
        for r in chosen:
            rows.append(dict(carID=car,timestamp=utc_string(start_us+delay_us+r['_source_us']-start['source_us']),
                             **{k:r[k] for k in COLUMNS[2:]}))
        plans.append(dict(carID=car,launch_delay_s=delay_us/1e6,launch_timestamp=utc_string(start_us+delay_us),
                          source_start=start,rows=len(chosen),duration_s=(chosen[-1]['_source_us']-start['source_us'])/1e6))
    rows.sort(key=lambda r:(r['timestamp'],r['carID']))
    return rows,plans


def generate(args):
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Output directory is not empty; use a new directory')
    route,starts,sources,boundaries=load_route(args.imu,args.vision,args.sessions,args.stop_speed,args.stop_seconds)
    origin=datetime.fromisoformat(args.start.replace('Z','+00:00'))
    if origin.tzinfo is None:raise ValueError('--start requires a timezone')
    origin=origin.astimezone(timezone.utc);delta=origin-datetime(1970,1,1,tzinfo=timezone.utc)
    start_us=(delta.days*86400+delta.seconds)*1000000+delta.microseconds
    args.output.mkdir(parents=True,exist_ok=True)
    summary=dict(run_id=args.run_id,synthetic=True,source='One Waterloo campus drive copied into virtual cars',
                 start=utc_string(start_us),columns=COLUMNS,
                 stop_threshold_mps=args.stop_speed,minimum_stationary_remaining_s=args.stop_seconds,
                 timestamp_semantics='Retrospective target-patch end time, shifted to the scenario clock; NOT inference arrival time',
                 yolo_semantics='Any detected pothole frame in the 160 ms patch. NULL means missing/insufficient video coverage; false requires observed coverage. Video alignment is approximate MP4 metadata.',
                 gps_policy='Only existing fresh GPS attached to finalized patches; stale fixes omitted, no interpolation',
                 route_policy='Each car follows the remaining recorded route to session5, preserving wall-clock gaps. No wraparound, invented paths, or speed scaling.',
                 source_sessions=sources,session_boundaries=boundaries,variants={})
    for variant in ('staggered','cascade'):
        rows,plans=materialize(route,starts,variant,args.run_id,start_us,args.cars,args.headway)
        with (args.output/f'{variant}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=COLUMNS);writer.writeheader()
            for row in rows:writer.writerow({k:('true' if v else 'false') if type(v) is bool else v for k,v in row.items()})
        with (args.output/f'{variant}.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        summary['variants'][variant]=dict(rows=len(rows),cars=plans,
            imu_positive_rows=sum(r['imu_defect_detected'] is True for r in rows),
            yolo_positive_rows=sum(r['yolo_pothole_detected'] is True for r in rows),
            yolo_unknown_rows=sum(r['yolo_pothole_detected'] is None for r in rows),
            csv_sha256=sha(args.output/f'{variant}.csv'))
    (args.output/'schema.sql').write_text(SCHEMA)
    # Relative paths intentionally make this portable after moving the bundle.
    (args.output/'load.sql').write_text('''\\set ON_ERROR_STOP on
BEGIN;
\\ir schema.sql
CREATE TEMP TABLE fleet_import (LIKE simulated_car_observations INCLUDING DEFAULTS) ON COMMIT DROP;
\\copy fleet_import FROM 'staggered.csv' WITH (FORMAT csv, HEADER true, NULL '')
\\copy fleet_import FROM 'cascade.csv' WITH (FORMAT csv, HEADER true, NULL '')
INSERT INTO simulated_car_observations SELECT * FROM fleet_import
ON CONFLICT ("carID", "timestamp") DO NOTHING;
COMMIT;
''')
    (args.output/'manifest.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (args.output/'README.md').write_text('''# Synthetic Waterloo fleet scenarios

These virtual cars replay ONE recorded drive. They are not independent observations or evidence of repeated road damage.

- `staggered.csv` / `.jsonl`: one car per session, all launched at the same scenario time from a measured stationary interval. Each follows the remaining sessions to the end.
- `cascade.csv` / `.jsonl`: identical route copies launched at the configured headway (default 20 seconds), all from the first selected stationary start.
- `manifest.json`: exact starts, measured stop intervals and speed, car IDs, source hashes, gaps, counts and timing semantics.
- `schema.sql` and `load.sql`: seven-column PostgreSQL schema and atomic, repeatable import into a separate `simulated_car_observations` table.

Rows follow finalized 160 ms IMU patches. Timestamp means road-observation/target-patch end time shifted onto the scenario clock, NOT the time inference first becomes available. GPS gaps and recording pauses remain gaps; the generator does not invent locations or wrap the end back to the beginning. Car IDs contain the run ID and variant so variants cannot collide. Use a NEW run ID for different inputs/settings; retrying the SAME bundle inserts no duplicates.

`imu_defect_detected` is the existing Kalman/hysteresis decision. `road_quality` is the calibrated good/medium/bad category. `yolo_pothole_detected` is true if any aligned frame in that patch detected a pothole; false requires observed negative coverage. Missing video coverage is NULL (empty CSV field), not false. Session1 has no YOLO export. Invalid IMU predictions similarly remain NULL. Camera/IMU alignment uses approximate MP4 creation metadata; coordinates are vehicle positions, not triangulated pothole positions. Both model outputs are predictions, not verified labels.

To import BOTH variants when ready, set DATABASE_URL privately, change into this bundle directory, and run:

```bash
psql "$DATABASE_URL" -f load.sql
```

Generating the bundle does not connect to or change Tiger. The table uses carID, timestamp, latitude, longitude, imu_defect_detected, yolo_pothole_detected, and road_quality. No existing inference or pothole tables are modified.
''')
    print(json.dumps(dict(output=str(args.output),starts=starts,variants={k:{'cars':len(v['cars']),'rows':v['rows'],'yolo_unknown_rows':v['yolo_unknown_rows']} for k,v in summary['variants'].items()}),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--imu',type=Path,required=True,help='Local IMU replay export directory')
    p.add_argument('--vision',type=Path,required=True,help='Export directory with optional sessionN/vision.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sessions',nargs='+',default=[f'session{i}' for i in range(1,6)])
    p.add_argument('--start',default=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),help='Scenario start, ISO 8601 with timezone')
    p.add_argument('--run-id',default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    p.add_argument('--cars',type=int,default=5,help='Cascade fleet size; staggered uses one car per session')
    p.add_argument('--headway',type=float,default=20,help='Cascade launch delay in seconds')
    p.add_argument('--stop-speed',type=float,default=.3,help='Maximum stationary speed, m/s')
    p.add_argument('--stop-seconds',type=float,default=2,help='Minimum measured stationary time after launch')
    args=p.parse_args()
    if (args.cars<1 or not np.isfinite([args.headway,args.stop_speed,args.stop_seconds]).all()
        or args.headway<0 or args.stop_speed<0 or args.stop_seconds<=0):p.error('Invalid fleet or stop parameters')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',args.run_id):p.error('--run-id must be 1..64 letters, numbers, _ or -')
    if not args.sessions or len(set(args.sessions))!=len(args.sessions):p.error('Sessions must be nonempty and unique')
    generate(args)


if __name__=='__main__':main()
