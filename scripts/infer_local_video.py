"""Run the selected YOLO26 checkpoint on every frame and attach replay outputs."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['YOLO_AUTOINSTALL']='false'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import torch
from ultralytics import YOLO, __version__ as ultralytics_version
from vision_inference.contract import filter_detections
from scripts.export_baseten_sessions import video_start_offset


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,allow_nan=False,indent=2));temp.replace(path)


def run_video(model,meta,video,folder,args,checkpoint_sha):
    started=time.monotonic();cap=cv2.VideoCapture(str(video))
    fps=cap.get(cv2.CAP_PROP_FPS);expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps<=0:raise ValueError(f'Cannot open {video}')
    rotation=cap.get(cv2.CAP_PROP_ORIENTATION_META)
    encoded_size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    if not cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1) and rotation:
        raise RuntimeError('Decoder cannot honor the video orientation metadata')
    frames_dir=folder/'frames';frames_dir.mkdir(exist_ok=True)
    origin_ms=int(meta['timestamp_origin_unix_ns'])/1e6
    offset=video_start_offset(video,origin_ms)
    classes={int(k):v for k,v in model.names.items()}
    frames=[];detections=[];index=0;last_time=-1.;fallback_count=0;batch=[];batch_meta=[]
    all_path=folder/'yolo_frames.jsonl.gz'
    def process(handle):
        nonlocal batch,batch_meta
        if not batch:return
        results=model.predict(batch,device=args.device,quantize=32,imgsz=640,conf=.4,verbose=False)
        for result,(idx,pts,frame) in zip(results,batch_meta):
            h,w=frame.shape[:2]
            raw=[dict(box=b[:4].tolist(),conf=float(b[4]),cls=int(b[5])) for b in result.boxes.data.cpu().numpy()]
            kept=filter_detections({'detections':raw},w,h,classes)
            row=dict(frame_index=idx,video_time_s=pts,imu_time_s_estimate=pts+offset if offset is not None else None,
                     width=w,height=h,detections=kept,provider='local_ultralytics',checkpoint_sha256=checkpoint_sha)
            handle.write(json.dumps(row,allow_nan=False)+'\n')
            # Preview images match exactly the boxes shown by the viewer.
            scale=min(1.,960/max(w,h));pw,ph=round(w*scale),round(h*scale)
            preview=cv2.resize(frame,(pw,ph)) if scale<1 else frame
            ok=cv2.imwrite(str(frames_dir/f'{idx:06d}.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,75])
            if not ok:raise RuntimeError('Could not save replay frame')
            boxes=[dict(d,box=[d['box'][0]*pw/w,d['box'][1]*ph/h,d['box'][2]*pw/w,d['box'][3]*ph/h]) for d in kept]
            frames.append(dict(frame_index=idx,video_time_s=pts,width=pw,height=ph,detections=boxes))
            for d in kept:
                detections.append(dict(frame_index=idx,video_time_s=pts,imu_time_s_estimate=row['imu_time_s_estimate'],
                                       label=d['label'],confidence=d['conf'],x1=d['box'][0],y1=d['box'][1],x2=d['box'][2],y2=d['box'][3],
                                       image_width=w,image_height=h))
        batch=[];batch_meta=[]
    with gzip.open(all_path,'wt') as handle:
        while True:
            ok,frame=cap.read()
            if not ok:break
            pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if not 0<=pts or (index and pts<=last_time):pts=index/fps;fallback_count+=1
            if pts<=last_time:raise ValueError('Video timestamps are not increasing')
            last_time=pts;batch.append(frame);batch_meta.append((index,pts,frame));index+=1
            if len(batch)==args.batch:
                process(handle)
                if index% (args.batch*20)==0:print(f'{meta["session_id"]} YOLO {index}/{expected} frames ({index/(time.monotonic()-started):.0f} fps)',flush=True)
        process(handle)
    cap.release()
    if index!=expected:raise RuntimeError(f'Incomplete decode: {index}/{expected}')
    fields=['frame_index','video_time_s','imu_time_s_estimate','label','confidence','x1','y1','x2','y2','image_width','image_height']
    with (folder/'yolo_detections.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(detections)
    vision=dict(provider='local_ultralytics',checkpoint_sha256=checkpoint_sha,classes=classes,
                fps=fps,duration_s=last_time+1/fps,video_offset_s=offset,
                alignment='mp4_creation_estimate' if offset is not None else 'unconfirmed',
                source_video=str(video.resolve()),source_video_sha256=sha(video),
                encoded_size=encoded_size,rotation_degrees=rotation,orientation_applied=True,
                box_coordinates='upright display image after MP4 rotation',
                source_frames=expected,processed_frames=index,frame_stride=1,
                confidence_threshold=.4,precision='float32',imgsz=640,
                inference_device=str(model.predictor.device),ultralytics_version=ultralytics_version,
                timestamp_fallback_frames=fallback_count,frames=frames)
    write(folder/'vision.json',vision)
    stats=dict(yolo_frames=index,yolo_detections=len(detections),yolo_positive_frames=sum(bool(f['detections']) for f in frames),
               yolo_inference_seconds=time.monotonic()-started,video_duration_s=vision['duration_s'],video_offset_s=offset,
               vision_available=True,yolo_frame_stride=1,yolo_fps=fps,yolo_checkpoint_sha256=checkpoint_sha,
               video_rotation_degrees=rotation,video_orientation_applied=True,
               source_video=str(video.resolve()),source_video_sha256=vision['source_video_sha256'])
    meta.update(stats);meta['display_name']=meta['session_id'].replace('session','Session ')+' · local IMU + YOLO'
    meta['outputs_sha256'].update({name:sha(folder/name) for name in ['yolo_frames.jsonl.gz','yolo_detections.csv','vision.json']})
    write(folder/'receipt.json',meta)
    print(json.dumps(dict(session=meta['session_id'],**stats)),flush=True)


def main(args):
    torch.set_num_threads(4);cv2.setNumThreads(4)
    checkpoint_sha=sha(args.checkpoint)
    model=YOLO(str(args.checkpoint)).to(args.device)
    if model.names!={0:'pothole'}:raise ValueError(f'Unexpected checkpoint classes: {model.names}')
    manifest=json.loads((args.output/'manifest.json').read_text())
    manifest['vision_checkpoint_sha256']=checkpoint_sha
    manifest['local_runtime']=dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,ultralytics=ultralytics_version)
    for meta in manifest['sessions']:
        if meta['session_id'] not in args.sessions:continue
        run_video(model,meta,args.input/(meta['session_id']+'_vid.mp4'),args.output/meta['session_id'],args,checkpoint_sha)
        write(args.output/'manifest.json',manifest)
    manifest['completed_at']=datetime.now(timezone.utc).isoformat();write(args.output/'manifest.json',manifest)
    write(args.output/'summary.json',[{k:s.get(k) for k in ['session_id','samples','duration_s','finalized_patches','quality_counts','disturbance_events','yolo_frames','yolo_positive_frames','yolo_detections','video_offset_s']} for s in manifest['sessions']])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--sessions',nargs='+',default=['session2','session3','session4','session5'])
    p.add_argument('--batch',type=int,default=32)
    main(p.parse_args())
