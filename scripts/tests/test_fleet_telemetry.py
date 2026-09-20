import csv,hashlib,importlib.util,json
from pathlib import Path
import pandas as pd
spec=importlib.util.spec_from_file_location('telemetry',Path(__file__).parents[1]/'import_fleet_telemetry.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_export_preserves_target_availability_gaps_nulls_and_cascade(tmp_path):
    source=tmp_path/'source';bundle=tmp_path/'bundle';bundle.mkdir();sessions=[]
    for name,origin in [('session2',1000000),('session3',3000000)]:
        folder=source/name;folder.mkdir(parents=True)
        pd.DataFrame([dict(time_s=.32,input_available_s=.325,accel_x=1.,accel_y=2.,accel_z=9.8,speed=float('nan'),settling=False)]).to_parquet(folder/'samples.parquet')
        row=dict(end_s=.32,start_s=.16,available_s=.64,target_patch=1,quality_probability=[.2,.7,.1],probability=.6,original_probability=.8,quality_name='medium',disturbance=True,valid=True,settling=False,ensemble_sha256='abc',calibration_id='cal')
        (folder/'final_predictions.jsonl').write_text(json.dumps(row)+'\n')
        sessions.append(dict(session=name,origin_unix_us=origin,samples_sha256=hashlib.sha256((folder/'samples.parquet').read_bytes()).hexdigest(),predictions_sha256=hashlib.sha256((folder/'final_predictions.jsonl').read_bytes()).hexdigest()))
    cars=[dict(carID=f'car-{i}',source_start=dict(source_us=1320000),duration_s=2,launch_timestamp=f'2026-09-20T12:00:{i*20:02d}Z') for i in range(2)]
    (bundle/'manifest.json').write_text(json.dumps(dict(source_sessions=sessions,variants=dict(cascade=dict(cars=cars)))))
    module.export(bundle,source)
    samples=list(csv.DictReader((bundle/'telemetry_samples.csv').open()));preds=list(csv.DictReader((bundle/'telemetry_predictions.csv').open()))
    assert len(samples)==len(preds)==4
    assert all(s['speed_mps']=='' for s in samples)
    assert module.unix(samples[1]['timestamp'])-module.unix(samples[0]['timestamp'])==2000000
    assert module.unix(samples[2]['timestamp'])-module.unix(samples[0]['timestamp'])==20000000
    assert module.unix(preds[0]['available_at'])-module.unix(preds[0]['timestamp'])==320000
    assert module.unix(samples[0]['available_at'])-module.unix(samples[0]['timestamp'])==5000
    assert preds[0]['defect_probability_kalman']=='0.6'
    assert preds[0]['imu_defect_detected']=='true'
    assert all(p['timestamp']==s['timestamp'] for p,s in zip(preds,samples))
