"""Build the video handoff from a fleet bundle and its frozen vision metadata."""
import argparse
import json
from pathlib import Path


def build(bundle, rendered):
    manifest=json.loads((bundle/'manifest.json').read_text())
    sessions=[]
    for s in manifest['source_sessions']:
        name=s['session']
        v=json.loads(Path(s['vision']['source']).read_text())
        current=json.loads((rendered/name/'vision.json').read_text())
        receipt=json.loads((rendered/name/'render_receipt.json').read_text())
        pts=[f['video_time_s'] for f in v['frames']]
        assert pts == [f['video_time_s'] for f in current['frames']]
        assert v['source_video_sha256']==current['source_video_sha256']
        assert v['checkpoint_sha256']==current['checkpoint_sha256']
        assert receipt['frames']==len(pts) and receipt['fps']==v['fps']
        assert all(f['frame_index']==i for i,f in enumerate(v['frames']))
        sessions.append(dict(session=name,origin_unix_us=s['origin_unix_us'],
            duration_s=s['duration_s'],video_offset_s=v['video_offset_s'],
            video_duration_s=v['duration_s'],fps=v['fps'],frame_pts_s=pts,
            url=f'/api/session/{name}/annotated.mp4',source_video_sha256=v['source_video_sha256']))
    result=dict(version=1,start=manifest['start'],sessions=sessions,
        variants={k:dict(cars=v['cars']) for k,v in manifest['variants'].items()},
        timing='IMU seconds = original video seconds + video_offset_s; preserve recording gaps',
        labels='Use database labels at target timestamps; burned-in IMU labels use delayed inference availability')
    (bundle/'video_sync.json').write_text(json.dumps(result,separators=(',',':')))
    print(f'Wrote {bundle}/video_sync.json for {len(sessions)} sessions')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('rendered',type=Path)
    a=p.parse_args();build(a.bundle,a.rendered)
