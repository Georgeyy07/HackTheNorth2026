"""Run original FP32 IMU ensemble and deployed post-processing on raw JSONL files."""
import argparse
import asyncio
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imu_inference.recording import read_recording
from imu_inference.model import EnsemblePredictor
from imu_inference.stream import RoadStream
from imu_inference.filters import DEFAULT_ALERT_FILTER

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LocalClient:
    def __init__(self,model):self.model=model
    async def predict(self,payload):return self.model.predict(payload)


def events_from(finals):
    events={}
    for r in finals:
        event_id=r.get('event_id')
        if event_id is None:continue
        if r.get('event_transition')=='start':
            events[event_id]=dict(event_id=event_id,start_s=r['start_s'],alert_available_s=r['available_s'],
                end_s=None,closed=False,censored=False,peak_defect_score=r['probability'],
                latitude=r['latitude'],longitude=r['longitude'])
        e=events.get(event_id)
        if e is None:continue
        if r['probability'] is not None:e['peak_defect_score']=max(e['peak_defect_score'],r['probability'])
        if r.get('event_transition') in ('end','censored'):
            e.update(end_s=r['start_s'] if r['event_transition']=='end' else None,closed=True,censored=r['event_transition']=='censored')
    for e in events.values():
        if not e['closed']:e['censored']=True
    return list(events.values())


async def infer(path,args,client,calibration):
    start=time.monotonic();name=path.stem;folder=args.output/name;folder.mkdir(parents=True,exist_ok=True)
    (samples,gps,_,_,origin_ns),metadata=read_recording(path)
    samples.to_parquet(folder/'samples.parquet',index=False);gps.to_parquet(folder/'gps_fixes.parquet',index=False)
    origin_ms=origin_ns/1e6
    availability=samples.input_available_s.to_numpy()
    fix_indices=np.searchsorted(gps.time_s.to_numpy(),availability,side='right')-1
    gps_rows=gps.to_dict('records');records=[];valid_fixes=[]
    for i,row in enumerate(samples.to_dict('records')):
        record=dict(time=row['time_s'],available_at_ms=origin_ms+row['input_available_s']*1000,settling=bool(row['settling']),
                    **{k:float(row[k]) if pd.notna(row[k]) else None for k in ('accel_x','accel_y','accel_z','speed')})
        j=int(fix_indices[i]);valid=False
        if j>=0:
            fix=gps_rows[j];now=record['available_at_ms']
            valid=0<=now-fix['source_timestamp_unix_ms']<=3000 and fix['received_at_unix_ms']<=now
            if valid:record.update(latitude=fix['latitude_deg'],longitude=fix['longitude_deg'],gps_timestamp_ms=fix['source_timestamp_unix_ms'],gps_received_at_ms=fix['received_at_unix_ms'])
        valid_fixes.append(j if valid else -1);records.append(record)
    stream=RoadStream(calibration);updates=[]
    for offset in range(0,len(records),1024):
        batch=await stream.push(records[offset:offset+1024],client)
        for row in batch:
            row['available_s']=max(row['available_s'],float(availability[row['emitted_after_samples']-1]))
            candidates=[i for i in range(row['target_sample_start'],row['target_sample_end']) if valid_fixes[i]>=0]
            i=candidates[-1] if candidates else None
            index=valid_fixes[i] if i is not None else None
            row.update(provider='local_pytorch',target_latitude_deg=row['latitude'],target_longitude_deg=row['longitude'],
                       target_gps_valid=row['latitude'] is not None,target_gps_fix_index=index,
                       target_gps_age_s=(records[i]['available_at_ms']-records[i]['gps_timestamp_ms'])/1000 if i is not None else None)
        updates.extend(batch)
        if offset%10240==0:print(f'{name}: {min(offset+1024,len(records))}/{len(records)} samples',flush=True)
    finals=[r for r in updates if r['is_final']]
    assert len(finals)==max(0,len(samples)//16-2)
    assert [r['target_patch'] for r in finals]==list(range(len(finals)))
    assert all(updates[i]['available_s']<=updates[i+1]['available_s'] for i in range(len(updates)-1))
    with gzip.open(folder/'updates.jsonl.gz','wt') as f:
        for r in updates:f.write(json.dumps(r,allow_nan=False)+'\n')
    with (folder/'final_predictions.jsonl').open('w') as f:
        for r in finals:f.write(json.dumps(r,allow_nan=False)+'\n')
    flat=[]
    for r in finals:
        q=r.get('quality_probability');raw=r.get('original_quality_probability')
        flat.append(dict(**{k:r.get(k) for k in ('target_patch','start_s','end_s','available_s','valid','quality_grade','quality_name','probability','original_probability','disturbance','event_id','event_transition','latitude','longitude','settling')},
                         roughness_score_0_100=50*(q[1]+2*q[2]) if q else None,
                         **{f'p_{label}':q[i] if q else None for i,label in enumerate(['good','medium','bad'])},
                         **{f'raw_p_{label}':raw[i] if raw else None for i,label in enumerate(['good','medium','bad'])}))
    pd.DataFrame(flat).to_csv(folder/'final_predictions.csv',index=False)
    pd.DataFrame(flat).to_parquet(folder/'final_predictions.parquet',index=False)
    events=events_from(finals);write(folder/'events.json',events)
    quality_counts={n:sum(r['quality_name']==n for r in finals) for n in ['good','medium','bad']}
    meta=dict(session_id=name,dataset='user_jsonl',split='user',display_name=name.replace('session','Session ')+' · local ensemble',
              samples=len(samples),duration_s=len(samples)/100,timestamp_origin_unix_ns=str(origin_ns),
              quality_mode='ordinal',quality_classes=['good','medium','bad'],inference_provider='local_pytorch',inference_status='complete',
              gyro_available=True,updates=len(updates),finalized_patches=len(finals),provisional_tail_patches=min(2,len(samples)//16),
              incomplete_tail_samples=len(samples)%16,invalid_finalized_patches=sum(not r['valid'] for r in finals),
              quality_counts=quality_counts,disturbance_events=len(events),alert_filter=DEFAULT_ALERT_FILTER,
              calibration=calibration,calibration_note='Saved car/mount bias reused; session5 was not part of the original calibration binding.' if name=='session5' else 'Saved car/mount bias applied once after consensus.',
              source_jsonl=str(path.resolve()),source_jsonl_sha256=sha(path),ensemble_sha256=client.model.ensemble_sha256,
              runtime_device=str(client.model.device),runtime_precision='float32',inference_seconds=time.monotonic()-start,
              outputs_sha256={n:sha(folder/n) for n in ['samples.parquet','updates.jsonl.gz','final_predictions.jsonl','events.json']},**metadata)
    write(folder/'receipt.json',meta);print(json.dumps(dict(session=name,finals=len(finals),quality=quality_counts,events=len(events),seconds=meta['inference_seconds'])),flush=True)
    return meta


async def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    calibration=json.loads((ROOT/'imu_inference/calibration_profile.json').read_text())
    client=LocalClient(EnsemblePredictor(ROOT/'models/ordinal_pvs',device=args.device))
    sessions=[]
    for name in args.sessions:
        sessions.append(await infer(args.input/(name+'.jsonl'),args,client,calibration))
        manifest=dict(sessions=sessions,samples=sum(s['samples'] for s in sessions),duration_s=sum(s['duration_s'] for s in sessions),
                      provider='local_pytorch',quality_mode='ordinal',score_filter_applied=True,alert_filter=DEFAULT_ALERT_FILTER,
                      calibration=calibration,created_at=datetime.now(timezone.utc).isoformat())
        write(args.output/'manifest.json',manifest)
    write(args.output/'summary.json',[{k:s[k] for k in ['session_id','samples','duration_s','finalized_patches','quality_counts','disturbance_events','inference_seconds']} for s in sessions])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--sessions',nargs='+',default=['session1','session2','session3','session4','session5'])
    asyncio.run(main(p.parse_args()))
