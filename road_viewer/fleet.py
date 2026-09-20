"""Read-only preview of generated local fleet files. No database access."""
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import Response, FileResponse


def install_fleet_routes(app, directory, here):
    if not directory:
        return False
    root=Path(directory).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    origin=datetime.fromisoformat(manifest['start'].replace('Z','+00:00'))

    @lru_cache(maxsize=2)
    def payload(variant):
        groups={car['carID']:dict(car,points=[]) for car in manifest['variants'][variant]['cars']}
        with (root/f'{variant}.jsonl').open() as f:
            for line in f:
                r=json.loads(line)
                t=(datetime.fromisoformat(r['timestamp'].replace('Z','+00:00'))-origin).total_seconds()
                groups[r['carID']]['points'].append([t,r['latitude'],r['longitude'],r['imu_defect_detected'],r['yolo_pothole_detected'],r['road_quality']])
        return json.dumps(dict(variant=variant,start=manifest['start'],synthetic=True,
            columns=['time_s','latitude','longitude','imu_defect_detected','yolo_pothole_detected','road_quality'],
            cars=list(groups.values()),duration_s=max(c['points'][-1][0] for c in groups.values()),
            stop_threshold_mps=manifest['stop_threshold_mps'],minimum_stop_s=manifest['minimum_stationary_remaining_s']),allow_nan=False).encode()

    @app.get('/api/fleet/{variant}')
    def fleet_data(variant:str):
        if variant not in ('staggered','cascade') or variant not in manifest['variants']:
            raise HTTPException(404,'Unknown fleet variant')
        return Response(payload(variant),media_type='application/json')

    @app.get('/api/fleet-sync')
    def fleet_sync():
        return FileResponse(root/'video_sync.json', media_type='application/json', headers={'Cache-Control':'no-store'})

    @app.get('/fleet')
    def fleet_page():
        return FileResponse(here/'static/fleet.html',headers={'Cache-Control':'no-store'})

    return True
