"""Build an isolated presentation fixture from real routing geometry and recorded detections."""
import argparse,json,hashlib,urllib.request
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from route_planner.navigation import project,lengths


def build(imu,vision,output,route_cache):
    route_cache.mkdir(parents=True,exist_ok=True)
    requests={
        'presentation-baseline.json':'-80.5438,43.4758;-80.53822,43.474662;-80.528,43.475',
        'presentation-routes.json':'-80.5438,43.4758;-80.528,43.475',
    }
    for filename,coordinates in requests.items():
        path=route_cache/filename
        if not path.is_file():
            url='https://router.project-osrm.org/route/v1/driving/'+coordinates+'?overview=full&geometries=geojson&steps=true&annotations=duration,distance&continue_straight=true'
            request=urllib.request.Request(url,headers={'User-Agent':'Roadscope-presentation/1.0'})
            with urllib.request.urlopen(request,timeout=20) as response:raw=json.load(response)
            if raw.get('code')!='Ok' or not raw.get('routes'):raise ValueError('Routing provider returned no route')
            path.write_text(json.dumps(raw))
    routes=[]
    for name,filename,label in [('recorded','presentation-baseline.json','Phillip → University'),('bypass','presentation-routes.json','Albert → Hickory → Hazel')]:
        raw=json.loads((route_cache/filename).read_text())['routes'][0]
        routes.append(dict(id=name,label=label,coords=[[y,x] for x,y in raw['geometry']['coordinates']],duration_s=raw['duration'],distance_m=raw['distance'],quality_regions=[]))
    folder=imu/'session3';records=[json.loads(l) for l in (folder/'final_predictions.jsonl').read_text().splitlines()]
    v=json.loads((vision/'session3/vision.json').read_text());positive=[f['video_time_s']+v['video_offset_s'] for f in v['frames'] if f['detections']]
    hazards=[]
    for r in records:
        if not r['latitude'] or not r['disturbance'] or not any(r['start_s']<=t<r['end_s'] for t in positive):continue
        point=[r['latitude'],r['longitude']];distance,along=project(point,routes[0]['coords'])
        if distance>25:continue
        if any(project(point,[h['coords'],h['coords']])[0]<25 for h in hazards):continue
        scout_start=70 if r['end_s']<150 else 150
        hazards.append(dict(id=f"session3-patch-{r['target_patch']}",coords=point,severity=1.,source_session='session3',
                            source_time_s=r['end_s'],source_available_s=r['available_s'],source_quality=r['quality_name'],
                            scout_id='scout-a' if scout_start==70 else 'scout-b',available_s=r['available_s']-scout_start,
                            provenance='recorded YOLO + IMU positive in same patch; discovery clock shifted for demo'))
        routes[0]['quality_regions'].append(dict(start_m=max(0,along-25),end_m=along+25,grade=r['quality_name'],available_s=r['available_s']-scout_start,provenance='recorded model classification'))
    # Only the bypass middle section receives a synthetic good-surface prior.
    routes[1]['quality_regions']=[dict(start_m=450,end_m=1600,grade='good',available_s=0,provenance='synthetic presentation assumption; not measured')]
    scouts=[]
    for name,start in [('scout-a',70),('scout-b',150)]:
        rows=[r for i,r in enumerate(records) if i%3==0 and start<=r['end_s']<=start+110 and r['latitude'] is not None]
        scouts.append(dict(id=name,label='Scout A' if name=='scout-a' else 'Scout B',source_session='session3',source_start_s=start,
                           points=[[round(r['end_s']-start,6),r['latitude'],r['longitude']] for r in rows]))
    result=dict(title='Columbia to MidCampus: fleet-aware navigation',origin_label='Columbia Street West, west of Phillip Street',
                destination_label='MidCampus Drive entrance at University Avenue West',routes=routes,hazards=hazards,scouts=scouts,
                default_route='recorded',source_predictions_sha256=hashlib.sha256((folder/'final_predictions.jsonl').read_bytes()).hexdigest(),
                source_vision_sha256=hashlib.sha256((vision/'session3/vision.json').read_bytes()).hexdigest(),
                assumptions=['Subject car motion and discovery times are simulated.',
                 'Bypass middle section is assumed good for presentation; it includes unrecorded Albert/Hickory/Hazel roads.',
                 'Road coordinates and initial duration estimates come from OSRM, not invented straight lines.',
                 'Detected locations are vehicle GPS; lane-level defect position is unknown.'],
                geometry_source='https://router.project-osrm.org/route/v1/driving/',demo_only=True)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,separators=(',',':')))
    print(dict(hazards=len(hazards),discovery_seconds=[h['available_s'] for h in hazards],duration_seconds=[r['duration_s'] for r in routes]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--imu',type=Path,required=True);p.add_argument('--vision',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--route-cache',type=Path,required=True)
    a=p.parse_args();build(a.imu,a.vision,a.output,a.route_cache)
