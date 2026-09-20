"""Export and atomically import source-aligned fleet telemetry. No model reruns."""
import argparse,csv,json,hashlib,math,os
from pathlib import Path
from datetime import datetime,timezone,timedelta

SAMPLE_COLS=['carID','timestamp','available_at','source_session','source_time_s','accel_x','accel_y','accel_z','speed_mps','settling']
PRED_COLS=['carID','timestamp','patch_start','available_at','source_session','source_time_s','target_patch','p_good','p_medium','p_bad','defect_probability_kalman','defect_probability_raw','road_quality','imu_defect_detected','valid','settling','ensemble_sha256','calibration_id']

def stamp(us):return (datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(microseconds=int(us))).isoformat()
def unix(s):return round(datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()*1e6)
def clean(x):
    if x is None:return ''
    if isinstance(x,bool):return 'true' if x else 'false'
    if isinstance(x,float) and not math.isfinite(x):return ''
    return x

def export(bundle,source):
    import pandas as pd
    m=json.loads((bundle/'manifest.json').read_text());loaded={}
    for s in m['source_sessions']:
        path=source/s['session']
        for filename,key in [('samples.parquet','samples_sha256'),('final_predictions.jsonl','predictions_sha256')]:
            assert hashlib.sha256((path/filename).read_bytes()).hexdigest()==s[key],filename
        loaded[s['session']]=(pd.read_parquet(path/'samples.parquet').to_dict('records'),[json.loads(x) for x in (path/'final_predictions.jsonl').read_text().splitlines()])
    counts={};paths={}
    for kind,cols in [('samples',SAMPLE_COLS),('predictions',PRED_COLS)]:
        out=bundle/f'telemetry_{kind}.csv';n=0
        with out.open('w') as f:
            w=csv.writer(f);w.writerow(cols)
            for variant in m['variants'].values():
                for car in variant['cars']:
                    start=car['source_start']['source_us'];end=start+round(car['duration_s']*1e6);shift=unix(car['launch_timestamp'])-start
                    for s in m['source_sessions']:
                        origin=s['origin_unix_us'];rows=loaded[s['session']][0 if kind=='samples' else 1]
                        for r in rows:
                            seconds=r['time_s'] if kind=='samples' else r['end_s'];src=origin+round(seconds*1e6)
                            if not start<=src<=end:continue
                            when=stamp(src+shift);avail=stamp(origin+shift+round(r['input_available_s' if kind=='samples' else 'available_s']*1e6))
                            if kind=='samples':
                                row=[car['carID'],when,avail,s['session'],seconds,r['accel_x'],r['accel_y'],r['accel_z'],r['speed'],bool(r['settling'])]
                            else:
                                q=r['quality_probability'] or [None]*3
                                assert not r['valid'] or (abs(sum(q)-1)<1e-6 and all(0<=p<=1 for p in q) and 0<=r['probability']<=1)
                                row=[car['carID'],when,stamp(origin+shift+round(r['start_s']*1e6)),avail,s['session'],seconds,r['target_patch'],*q,r['probability'],r['original_probability'],r['quality_name'],r['disturbance'],r['valid'],r['settling'],r['ensemble_sha256'],r['calibration_id']]
                            w.writerow([clean(x) for x in row]);n+=1
        counts[kind]=n;paths[kind]=hashlib.sha256(out.read_bytes()).hexdigest()
    (bundle/'telemetry_export.json').write_text(json.dumps(dict(rows=counts,sha256=paths),indent=2));print(counts,flush=True)

SCHEMA='''
CREATE TABLE IF NOT EXISTS public.simulated_car_imu_samples (
 "carID" text NOT NULL,"timestamp" timestamptz NOT NULL,available_at timestamptz NOT NULL,
 source_session text NOT NULL CHECK(source_session IN ('session2','session3','session4','session5')),
 source_time_s double precision NOT NULL,accel_x double precision,accel_y double precision,accel_z double precision,
 speed_mps double precision,settling boolean NOT NULL,PRIMARY KEY("carID","timestamp"));
CREATE TABLE IF NOT EXISTS public.simulated_car_inference (
 "carID" text NOT NULL,"timestamp" timestamptz NOT NULL,patch_start timestamptz NOT NULL,available_at timestamptz NOT NULL,
 source_session text NOT NULL CHECK(source_session IN ('session2','session3','session4','session5')),
 source_time_s double precision NOT NULL,target_patch integer NOT NULL,
 p_good double precision CHECK(p_good BETWEEN 0 AND 1),p_medium double precision CHECK(p_medium BETWEEN 0 AND 1),
 p_bad double precision CHECK(p_bad BETWEEN 0 AND 1),defect_probability_kalman double precision CHECK(defect_probability_kalman BETWEEN 0 AND 1),
 defect_probability_raw double precision CHECK(defect_probability_raw BETWEEN 0 AND 1),
 road_quality text CHECK(road_quality IN ('good','medium','bad')),imu_defect_detected boolean,valid boolean NOT NULL,settling boolean NOT NULL,
 ensemble_sha256 text NOT NULL,calibration_id text,PRIMARY KEY("carID","timestamp"),
 CHECK(abs(p_good+p_medium+p_bad-1)<0.00001),CHECK(available_at >= "timestamp"));
CREATE INDEX IF NOT EXISTS simulated_car_inference_available_idx ON public.simulated_car_inference("carID",available_at);
CREATE INDEX IF NOT EXISTS simulated_car_imu_available_idx ON public.simulated_car_imu_samples("carID",available_at);
CREATE OR REPLACE VIEW public.simulated_car_observations_enriched AS
 SELECT o.*,i.p_good,i.p_medium,i.p_bad,i.defect_probability_kalman,i.defect_probability_raw,
 i.patch_start,i.available_at AS inference_available_at,i.source_session,i.source_time_s,
 s."timestamp" AS imu_sample_timestamp,s.accel_x,s.accel_y,s.accel_z,s.speed_mps,s.settling
 FROM public.simulated_car_observations o
 LEFT JOIN public.simulated_car_inference i USING("carID","timestamp")
 LEFT JOIN LATERAL (SELECT a.* FROM public.simulated_car_imu_samples a
 WHERE a."carID"=o."carID" AND a."timestamp" <= o."timestamp"
 AND a."timestamp" >= o."timestamp" - interval '20 milliseconds'
 ORDER BY a."timestamp" DESC LIMIT 1) s ON true;
'''

def ingest(bundle):
    import psycopg2
    report=json.loads((bundle/'telemetry_export.json').read_text());m=json.loads((bundle/'manifest.json').read_text())
    ids=[c['carID'] for v in m['variants'].values() for c in v['cars']]
    url=os.environ.get('DATABASE_URL') or os.environ['TIGER_DATA_URL'];stats={}
    with psycopg2.connect(url,connect_timeout=15) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='10s'; SET LOCAL statement_timeout='300s'")
            cur.execute(SCHEMA)
            for kind,table,cols in [('samples','simulated_car_imu_samples',SAMPLE_COLS),('predictions','simulated_car_inference',PRED_COLS)]:
                path=bundle/f'telemetry_{kind}.csv'
                assert hashlib.sha256(path.read_bytes()).hexdigest()==report['sha256'][kind]
                cur.execute(f'CREATE TEMP TABLE stage_{kind} (LIKE public.{table} INCLUDING ALL) ON COMMIT DROP')
                with path.open() as f:cur.copy_expert(f"COPY stage_{kind} FROM STDIN WITH (FORMAT CSV,HEADER TRUE,NULL '')",f)
                cur.execute(f'SELECT count(*) FROM stage_{kind}');assert cur.fetchone()[0]==report['rows'][kind]
                cur.execute(f'INSERT INTO public.{table} SELECT * FROM stage_{kind} ON CONFLICT ("carID","timestamp") DO NOTHING')
                cur.execute(f'SELECT count(*) FROM public.{table} d JOIN stage_{kind} s USING("carID","timestamp") WHERE d IS DISTINCT FROM s');assert cur.fetchone()[0]==0,'Existing telemetry differs'
                cur.execute(f'SELECT count(*) FROM public.{table} WHERE "carID"=ANY(%s)',(ids,));stats[kind]=cur.fetchone()[0];assert stats[kind]==report['rows'][kind]
            cur.execute('''SELECT count(*) FROM public.simulated_car_observations o LEFT JOIN public.simulated_car_inference i USING("carID","timestamp")
             WHERE o."carID"=ANY(%s) AND (i."carID" IS NULL OR o.road_quality IS DISTINCT FROM i.road_quality OR o.imu_defect_detected IS DISTINCT FROM i.imu_defect_detected)''',(ids,))
            assert cur.fetchone()[0]==0,'Map labels/timestamps differ'
    with psycopg2.connect(url,connect_timeout=15) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT count(*),count(p_good),count(accel_x) FROM public.simulated_car_observations_enriched WHERE "carID"=ANY(%s)',(ids,));stats['map_rows_matched']=cur.fetchone()
            assert stats['map_rows_matched'][0]==stats['map_rows_matched'][1]==stats['map_rows_matched'][2]
    stats['verified_at']=datetime.now(timezone.utc).isoformat()
    (bundle/'telemetry_import_receipt.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('--source',type=Path);p.add_argument('--import-only',action='store_true');a=p.parse_args()
    if a.import_only:ingest(a.bundle)
    else:export(a.bundle,a.source)
