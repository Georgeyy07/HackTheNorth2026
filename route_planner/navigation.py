"""Route-aware warnings and bounded, evidence-driven rerouting decisions.

Coordinates are [latitude, longitude]. Route geometry is supplied by a routing
provider; missing hazard reports never certify a road as safe.
"""
import math
from alert_service.alert_math import haversine_distance_m


def lengths(coords):
    out=[0.]
    for a,b in zip(coords,coords[1:]):out.append(out[-1]+haversine_distance_m(*a,*b))
    return out


def project(point,coords):
    cumulative=lengths(coords);best=(float('inf'),0.)
    sx=111320*math.cos(math.radians(point[0]));sy=111320
    for i,(a,b) in enumerate(zip(coords,coords[1:])):
        ax=(a[1]-point[1])*sx;ay=(a[0]-point[0])*sy
        dx=(b[1]-a[1])*sx;dy=(b[0]-a[0])*sy
        f=max(0,min(1,-(ax*dx+ay*dy)/(dx*dx+dy*dy))) if dx*dx+dy*dy else 0
        distance=math.hypot(ax+f*dx,ay+f*dy)
        if distance<best[0]:best=(distance,cumulative[i]+f*(cumulative[i+1]-cumulative[i]))
    return best


def position(route,progress):
    coords=route['coords'];ls=lengths(coords);progress=max(0,min(ls[-1],progress))
    for i in range(1,len(ls)):
        if ls[i]>=progress:
            f=(progress-ls[i-1])/(ls[i]-ls[i-1]) if ls[i]>ls[i-1] else 0
            return [coords[i-1][j]+f*(coords[i][j]-coords[i-1][j]) for j in range(2)]
    return coords[-1]


def remaining_metrics(route,progress,hazards,now):
    total=lengths(route['coords'])[-1];encountered=[]
    for h in hazards:
        if h['available_s']>now:continue
        distance,along=project(h['coords'],route['coords'])
        if distance<=25 and progress<along<=total:
            # Multiple patches/scouts for one nearby location are one hazard.
            if not any(abs(along-e['along_m'])<25 for e in encountered):
                encountered.append(dict(id=h['id'],along_m=along,distance_ahead_m=along-progress,severity=h.get('severity',1)))
    remaining=max(0,total-progress);eta=route['duration_s']*remaining/total
    bad=medium=good=0.
    for region in route.get('quality_regions',[]):
        if region['available_s']>now:continue
        covered=max(0,min(total,region['end_m'])-max(progress,region['start_m']))
        if region['grade']=='bad':bad+=covered
        elif region['grade']=='medium':medium+=covered
        elif region['grade']=='good':good+=covered
    penalty=45*sum(h['severity'] for h in encountered)+.08*bad+.025*medium
    return dict(route_id=route['id'],remaining_m=remaining,eta_s=eta,potholes=len(encountered),hazards=encountered,
                bad_m=bad,medium_m=medium,good_m=good,risk_penalty_s=penalty,utility_cost_s=eta+penalty)


def evaluate(routes,current_id,progress,hazards,now,last_reroute_s=-1e9):
    current=next(r for r in routes if r['id']==current_id);point=position(current,progress)
    baseline=remaining_metrics(current,progress,hazards,now);options=[]
    for route in routes:
        if route['id']==current_id:continue
        if haversine_distance_m(*route['coords'][-1],*current['coords'][-1])>15:continue
        lateral,candidate_progress=project(point,route['coords'])
        if lateral>10:continue  # Never teleport to a parallel road or invent a connector.
        candidate=remaining_metrics(route,candidate_progress,hazards,now)
        delta=candidate['eta_s']-baseline['eta_s']
        reduction=baseline['risk_penalty_s']-candidate['risk_penalty_s']
        gain=baseline['utility_cost_s']-candidate['utility_cost_s']
        # Give the driver at least 30 metres before the paths separate.
        ahead=position(current,min(lengths(current['coords'])[-1],progress+30))
        ahead_distance,ahead_progress=project(ahead,route['coords'])
        turn_room=ahead_distance<=10 and ahead_progress>=candidate_progress+15
        eligible=(baseline['remaining_m']>=150 and baseline['risk_penalty_s']>0 and reduction>=.4*baseline['risk_penalty_s'] and
                  gain>=15 and delta<=min(30,.1*baseline['eta_s']) and turn_room and now-last_reroute_s>=30)
        options.append(dict(**candidate,eta_delta_s=delta,utility_gain_s=gain,risk_reduction_s=reduction,
                            eligible=eligible,progress_m=candidate_progress))
    best=min((c for c in options if c['eligible']),key=lambda c:c['utility_cost_s'],default=None)
    return dict(current=baseline,alternatives=options,reroute=best,
                policy=dict(max_extra_seconds=30,max_extra_fraction=.1,min_risk_reduction_fraction=.4,
                            min_utility_gain_s=15,cooldown_s=30,warning_distance_m=200))


def warnings(route,progress,hazards,now,already_warned,speed_mps):
    if speed_mps<.5:return []
    found=[]
    for h in remaining_metrics(route,progress,hazards,now)['hazards']:
        if h['distance_ahead_m']<=200 and h['id'] not in already_warned:
            found.append(dict(type='warning',hazard_id=h['id'],distance_m=h['distance_ahead_m'],
                              message=f"Caution. Pothole ahead in {round(h['distance_ahead_m']/10)*10} metres."))
    return found
