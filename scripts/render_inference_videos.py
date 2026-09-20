"""Render upright H.264 videos from saved YOLO results without rerunning models."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import imageio_ffmpeg


def render(root, session):
    folder=root/session
    vision=json.loads((folder/'vision.json').read_text())
    if not vision.get('orientation_applied'):raise ValueError('Complete upright inference before rendering')
    with gzip.open(folder/'yolo_frames.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
    with (folder/'final_predictions.jsonl').open() as f:imu=[json.loads(x) for x in f]
    source=Path(vision['source_video']);cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
    width,height=rows[0]['width'],rows[0]['height'];fps=vision['fps'];output=folder/'annotated.mp4'
    temp=folder/'annotated.tmp.mp4';started=time.monotonic();imu_index=-1
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
             '-f','rawvideo','-pixel_format','bgr24','-video_size',f'{width}x{height}','-framerate',str(fps),'-i','pipe:0',
             '-i',str(source),'-map','0:v:0','-map','1:a?','-map_metadata','-1',
             '-c:v','libx264','-preset','veryfast','-crf','20','-threads','4','-pix_fmt','yuv420p',
             '-c:a','aac','-b:a','128k','-metadata:s:v:0','rotate=0','-movflags','+faststart',str(temp)]
    with (folder/'render.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for index,row in enumerate(rows):
                ok,frame=cap.read()
                if not ok or frame.shape[:2]!=(height,width):raise RuntimeError('Source frame coverage/orientation mismatch')
                for d in row['detections']:
                    x1,y1,x2,y2=map(lambda x:int(round(x)),d['box'])
                    color=(100,70,255);cv2.rectangle(frame,(x1,y1),(x2,y2),color,4)
                    label=f"{d['label']} {d['conf']:.0%}"
                    (tw,th),baseline=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.85,2)
                    top=max(0,y1-th-14);left=min(x1,max(0,width-tw-14))
                    cv2.rectangle(frame,(left,top),(left+tw+12,top+th+12),color,-1)
                    cv2.putText(frame,label,(left+6,top+th+4),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2,cv2.LINE_AA)
                # IMU is an independent estimate; synchronization is approximate.
                t=row['video_time_s'];mapped=t+vision['video_offset_s'] if vision['video_offset_s'] is not None else None
                if mapped is not None:
                    while imu_index+1<len(imu) and imu[imu_index+1]['available_s']<=mapped:imu_index+=1
                current=imu[imu_index] if imu_index>=0 and imu[imu_index]['valid'] else None
                cv2.rectangle(frame,(12,12),(width-12,116),(25,32,28),-1)
                cv2.putText(frame,f'{session} | {int(t//60):02d}:{t%60:05.2f} | YOLO26 pothole >=40%',(26,48),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2,cv2.LINE_AA)
                label=(f"IMU: {current['quality_name'].upper()} | defect score {current['probability']:.0%} | approx. sync" if current else 'IMU: waiting for finalized context')
                cv2.putText(frame,label,(26,89),cv2.FONT_HERSHEY_SIMPLEX,.7,(210,235,218),2,cv2.LINE_AA)
                process.stdin.write(frame.tobytes())
                if index and index%6000==0:print(f'{session} rendered {index}/{len(rows)}',flush=True)
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError(f'Video encoding failed; see {folder}/render.log')
        except BaseException:
            process.kill();process.wait();raise
        finally:cap.release()
    temp.replace(output)
    check=cv2.VideoCapture(str(output));count=int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_size=[int(check.get(cv2.CAP_PROP_FRAME_WIDTH)),int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))];check.release()
    if count!=len(rows) or actual_size!=[width,height]:raise RuntimeError('Rendered video frame count or size mismatch')
    result=dict(session=session,path=str(output),frames=count,width=width,height=height,fps=fps,codec='H.264',
                orientation='upright',bytes=output.stat().st_size,render_seconds=time.monotonic()-started)
    (folder/'render_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();cv2.setNumThreads(1)
    manifest=json.loads((args.output/'manifest.json').read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results=list(executor.map(lambda s:render(args.output,s['session_id']),manifest['sessions']))
    for s in manifest['sessions']:
        s['annotated_video_available']=True
        receipt=json.loads((args.output/s['session_id']/'receipt.json').read_text());receipt['annotated_video_available']=True
        (args.output/s['session_id']/'receipt.json').write_text(json.dumps(receipt,indent=2))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.output/'render_summary.json').write_text(json.dumps(results,indent=2))
