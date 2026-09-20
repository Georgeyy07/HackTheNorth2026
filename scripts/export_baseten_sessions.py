"""Recompute prepared phone recordings on Baseten and export an auditable replay.

No existing predictions are reused. Cloud responses are cached by request hash,
so interrupted runs resume without paying for successful requests again.
"""
import argparse
import asyncio
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imu_inference.client import BasetenClient, ENSEMBLE_SHA
from imu_inference.stream import RoadStream
from imu_inference.store import PredictionStore
from vision_inference.client import VisionClient
from vision_inference.service import MANIFEST

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, allow_nan=False))
    temp.replace(path)


async def retry(call):
    for attempt in range(5):
        try: return await call()
        except Exception as e:
            # Configuration/authentication errors cannot be fixed by retrying.
            response = getattr(e, 'response', None)
            if response is not None and 400 <= response.status_code < 500 and response.status_code != 429:
                raise RuntimeError(f'Baseten HTTP {response.status_code}: {response.text[:500]}') from e
            if attempt == 4: raise
            print(f'retrying {type(e).__name__}, attempt {attempt+2}', flush=True)
            await asyncio.sleep(min(20, 2**attempt))


def video_start_offset(video, origin_ms):
    """Estimate start alignment from MP4 creation time (whole-second precision).

    Reject large offsets rather than silently matching unrelated clocks.
    This is an estimate, not a measured camera/IMU synchronization signal.
    """
    with video.open('rb') as handle:
        def scan(end):
            while handle.tell()+8 <= end:
                start = handle.tell()
                size, kind = struct.unpack('>I4s', handle.read(8))
                if size == 1: size = struct.unpack('>Q', handle.read(8))[0]
                if size < 8 or start+size > end: return None
                if kind == b'moov':
                    result = scan(start+size)
                    if result is not None: return result
                elif kind == b'mvhd':
                    data = handle.read(min(size-8, 32))
                    if len(data) < 12: return None
                    created = int.from_bytes(data[4:12] if data[0] else data[4:8], 'big')
                    epoch = datetime(1904,1,1,tzinfo=timezone.utc).timestamp()
                    offset = epoch+created-origin_ms/1000
                    return offset if abs(offset) < 5 else None
                handle.seek(start+size)
        return scan(video.stat().st_size)


class CachedIMU:
    def __init__(self, client, folder):
        self.client, self.folder = client, folder
        self.requests = self.cache_hits = 0

    async def predict(self, payload):
        key = hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()
        path = self.folder / (key+'.json')
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())
        result = await retry(lambda: self.client.predict(payload))
        write_json(path, result)
        self.requests += 1
        return result


def gps_for(gps, t, origin_ms):
    i = int(np.searchsorted(gps.time_s.to_numpy(), t, side='right'))-1
    if i < 0: return i, {}
    fix = gps.iloc[i]
    now = origin_ms+t*1000
    if not 0 <= now-fix.source_timestamp_unix_ms <= 3000 or fix.received_at_unix_ms > now:
        return i, {}
    return i, dict(latitude=float(fix.latitude_deg), longitude=float(fix.longitude_deg),
                   gps_timestamp_ms=float(fix.source_timestamp_unix_ms), gps_received_at_ms=float(fix.received_at_unix_ms))


async def export_imu(meta, args, folder, client):
    name = meta['session_id']
    source = args.prepared/name
    samples = pd.read_parquet(source/'samples.parquet')
    gps = pd.read_parquet(source/'gps_fixes.parquet')
    calibration = json.loads((ROOT/'imu_inference/calibration_profile.json').read_text())
    stream = RoadStream(calibration)
    cache_dir = folder/'imu_cache'; cache_dir.mkdir(exist_ok=True)
    cached = CachedIMU(client, cache_dir)
    origin_ms = int(meta['timestamp_origin_unix_ns'])/1e6
    records = []
    for row in samples.to_dict('records'):
        _, fix = gps_for(gps, row['input_available_s'], origin_ms)
        records.append(dict(time=row['time_s'], available_at_ms=origin_ms+row['input_available_s']*1000,
                            settling=bool(row.get('settling',False)), **fix,
                            **{k:float(row[k]) if pd.notna(row[k]) else None for k in ('accel_x','accel_y','accel_z','speed')}))
    updates=[]
    for start in range(0,len(records),1024):
        batch=await stream.push(records[start:start+1024],cached)
        for row in batch:
            end=row['emitted_after_samples']-1
            row['available_s']=max(row['available_s'],float(samples.iloc[end].input_available_s))
            target_end=row['target_sample_end']-1
            fix_idx, fix=gps_for(gps,float(samples.iloc[target_end].input_available_s),origin_ms)
            row.update(target_latitude_deg=row['latitude'],target_longitude_deg=row['longitude'],
                       target_gps_valid=row['latitude'] is not None,
                       target_gps_fix_index=fix_idx if row['latitude'] is not None else None,
                       target_gps_age_s=(origin_ms+float(samples.iloc[target_end].input_available_s)*1000-fix['gps_timestamp_ms'])/1000 if fix else None,
                       score_kind='kalman_probability',provider='baseten')
        updates.extend(batch)
        if start%5120==0: print(f'{name} IMU {min(start+1024,len(records))}/{len(records)}',flush=True)
    with gzip.open(folder/'updates.jsonl.gz','wt') as f:
        for row in updates:f.write(json.dumps(row,allow_nan=False)+'\n')
    finals=[r for r in updates if r['is_final']]
    sid=str(uuid.uuid5(uuid.NAMESPACE_URL,str(folder.resolve())+ENSEMBLE_SHA))
    if args.database_url:
        store=PredictionStore(args.database_url)
        with store.connection() as conn:
            store.execute(conn,'INSERT INTO imu_sessions VALUES (?, ?, ?) ON CONFLICT (session_id) DO NOTHING',
                          (sid,datetime.now(timezone.utc).isoformat(),json.dumps(dict(client='baseten-session-replay',recording=name,ensemble_sha256=ENSEMBLE_SHA,calibration=calibration))))
        store.save(sid,finals)
    for file in ['samples.parquet','gps_fixes.parquet']:shutil.copy2(source/file,folder/file)
    return dict(imu_session_id=sid,updates=len(updates),finalized_patches=len(finals),
                cloud_requests=cached.requests,cached_requests=cached.cache_hits,
                quality_counts={label:sum(r['quality_name']==label for r in finals) for label in ['good','medium','bad']},
                disturbance_events=sum(r['event_transition']=='start' for r in finals),
                calibration_id=calibration['id'],calibration_offset=calibration['offset'])


async def export_video(meta,args,folder,client):
    import cv2
    name=meta['session_id'];video=args.input/(name+'_vid.mp4')
    cap=cv2.VideoCapture(str(video))
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<=0:raise ValueError('Cannot read video')
    duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    cache=folder/'vision_cache';cache.mkdir(exist_ok=True)
    images=folder/'frames';images.mkdir(exist_ok=True)
    semaphore=asyncio.Semaphore(4)
    result_rows=[]
    # Decode each requested frame once. OpenCV applies the MP4 display rotation.
    async def predict(t,data,width,height,index):
        async with semaphore:
            image_sha=hashlib.sha256(data).hexdigest()
            path=cache/(image_sha+'.json')
            if path.exists():result=json.loads(path.read_text())
            else:
                result=await retry(lambda:client.predict(data,width,height))
                write_json(path,result)
            result_rows.append(dict(video_time_s=t,frame_index=index,image=f'{index:06d}.jpg',
                                    width=width,height=height,detections=result['detections'],
                                    provider=result['provider'],checkpoint_sha256=result['checkpoint_sha256']))
    tasks=[]
    for index,t in enumerate(np.arange(0,duration,1/args.yolo_fps)):
        cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000)
        ok,frame=cap.read()
        if not ok:raise RuntimeError(f'Cannot decode {name} at {t}')
        h,w=frame.shape[:2]
        if max(h,w)>1280:frame=cv2.resize(frame,(round(w*1280/max(h,w)),round(h*1280/max(h,w))))
        h,w=frame.shape[:2];ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok:raise RuntimeError('JPEG encode failed')
        data=encoded.tobytes();(images/f'{index:06d}.jpg').write_bytes(data)
        tasks.append(asyncio.create_task(predict(float(t),data,w,h,index)))
        if len(tasks)==12:
            await asyncio.gather(*tasks);tasks=[]
            print(f'{name} YOLO {t:.0f}/{duration:.0f}s',flush=True)
    await asyncio.gather(*tasks);cap.release()
    result_rows.sort(key=lambda r:r['video_time_s'])
    # Unconfirmed camera timing must not create falsely located pothole reports.
    offset=video_start_offset(video,int(meta['timestamp_origin_unix_ns'])/1e6)
    result=dict(fps=args.yolo_fps,duration_s=duration,video_offset_s=offset,
                alignment='mp4_creation_estimate' if offset is not None else 'unconfirmed',
                timeline='video seconds; IMU time = video time + video_offset_s',frames=result_rows)
    write_json(folder/'vision.json',result)
    return dict(video_duration_s=duration,yolo_frames=len(result_rows),
                yolo_positive_frames=sum(bool(r['detections']) for r in result_rows),
                yolo_detections=sum(len(r['detections']) for r in result_rows),video_sha256=digest(video))


async def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    prepared=json.loads((args.prepared/'manifest.json').read_text())
    imu_info=json.loads((ROOT/'baseten/deployment.json').read_text())
    yolo_info=json.loads((ROOT/'baseten/yolo26/deployment.json').read_text())
    key=os.environ['BASETEN_API_KEY']
    imu=BasetenClient(imu_info['predict_url'],key)
    vision=VisionClient(yolo_info['predict_url'],key,classes={int(k):v for k,v in MANIFEST['classes'].items()},checkpoint_sha256=MANIFEST['checkpoint_sha256'])
    all_meta=[]
    try:
        for name in args.sessions:
            meta=next(s for s in prepared['sessions'] if s['session_id']==name)
            original=json.loads(Path(prepared['sources'][name]['export_manifest']).read_text())
            source_sha=digest(args.input/(name+'.jsonl'))
            if original['source_jsonl_sha256']!=source_sha:raise ValueError(f'{name}: prepared samples belong to another recording')
            folder=args.output/name;folder.mkdir(exist_ok=True)
            imu_stats,video_stats=await asyncio.gather(export_imu(meta,args,folder,imu),export_video(meta,args,folder,vision))
            result=dict(meta,display_name=f'{name.replace("session","Session ")} · Baseten',inference_provider='baseten',inference_status='complete',
                        quality_mode='ordinal',quality_classes=['good','medium','bad'],source_jsonl_sha256=source_sha,
                        imu_deployment=imu_info['deployment_id'],yolo_deployment=yolo_info['deployment_id'],
                        yolo_fps=args.yolo_fps,vision_available=True,**imu_stats,**video_stats)
            write_json(folder/'receipt.json',result);all_meta.append(result)
            write_json(args.output/'manifest.json',dict(sessions=all_meta,samples=sum(s['samples'] for s in all_meta),
                       duration_s=sum(s['duration_s'] for s in all_meta),provider='baseten',
                       imu=imu_info,yolo=yolo_info,quality_mode='ordinal',score_filter_applied=True))
            print(json.dumps(dict(session=name,**imu_stats,**video_stats)),flush=True)
    finally:await imu.close();await vision.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sessions',nargs='+',default=['session1','session2','session3','session4'])
    p.add_argument('--yolo-fps',type=float,default=1)
    args=p.parse_args()
    if not 0<args.yolo_fps<=10:p.error('--yolo-fps must be in (0,10]')
    args.database_url=os.environ.get('DATABASE_URL')
    asyncio.run(main(args))
