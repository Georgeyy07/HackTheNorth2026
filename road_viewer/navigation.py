"""Presentation replay and reusable navigation policy API. No database writes."""
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal
from fastapi import HTTPException,Query
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field,field_validator,model_validator
from route_planner.navigation import evaluate,warnings,lengths,position,project

class QualityRegion(BaseModel):
    start_m:float=Field(ge=0,allow_inf_nan=False)
    end_m:float=Field(gt=0,allow_inf_nan=False)
    grade:Literal['good','medium','bad']
    available_s:float=Field(ge=0,allow_inf_nan=False)
    provenance:str
    @model_validator(mode='after')
    def order(self):
        if self.end_m<=self.start_m:raise ValueError('Quality interval must be positive')
        return self

class Route(BaseModel):
    id:str=Field(min_length=1,max_length=80)
    coords:list[list[float]]=Field(min_length=2,max_length=2000)
    duration_s:float=Field(gt=0,le=86400,allow_inf_nan=False)
    quality_regions:list[QualityRegion]=Field(default_factory=list,max_length=100)
    @field_validator('coords')
    @classmethod
    def coordinates(cls,coords):
        for c in coords:
            if len(c)!=2 or not -90<=c[0]<=90 or not -180<=c[1]<=180:raise ValueError('Expected finite [lat, lon]')
        if lengths(coords)[-1]<=0:raise ValueError('Empty route')
        return coords
    @model_validator(mode='after')
    def disjoint_quality(self):
        regions=sorted(self.quality_regions,key=lambda q:q.start_m)
        if any(a.end_m>b.start_m for a,b in zip(regions,regions[1:])):raise ValueError('Quality regions overlap')
        return self

class Hazard(BaseModel):
    id:str=Field(min_length=1,max_length=100)
    coords:list[float]=Field(min_length=2,max_length=2)
    severity:float=Field(default=1,ge=0,le=1,allow_inf_nan=False)
    available_s:float=Field(ge=0,allow_inf_nan=False)
    @field_validator('coords')
    @classmethod
    def coordinates(cls,c):
        if not -90<=c[0]<=90 or not -180<=c[1]<=180:raise ValueError('Invalid coordinates')
        return c

class NavigationRequest(BaseModel):
    routes:list[Route]=Field(min_length=1,max_length=5)
    current_route_id:str
    location:list[float]=Field(min_length=2,max_length=2)
    speed_mps:float=Field(ge=0,le=70,allow_inf_nan=False)
    now_s:float=Field(ge=0,allow_inf_nan=False)
    hazards:list[Hazard]=Field(default_factory=list,max_length=100)
    last_reroute_s:float=Field(default=-1e9,allow_inf_nan=False)
    warned_ids:list[str]=Field(default_factory=list,max_length=1000)
    @field_validator('location')
    @classmethod
    def coordinates(cls,c):return Hazard.coordinates(c)
    @model_validator(mode='after')
    def identifiers(self):
        ids=[r.id for r in self.routes]
        if len(set(ids))!=len(ids) or self.current_route_id not in ids:raise ValueError('Invalid route IDs')
        if len({h.id for h in self.hazards})!=len(self.hazards):raise ValueError('Repeated hazard IDs')
        return self


def simulate(fixture,mode):
    routes=fixture['routes'];current=routes[0];progress=0.;now=0.;last=-1e9;warned=set();events=[];states=[];announced=set()
    # 250 ms replay ticks; alerts are emitted at the first tick <=200 m ahead.
    while now<=300:
        for h in fixture['hazards']:
            if h['available_s']<=now and h['id'] not in announced:
                announced.add(h['id']);events.append(dict(type='discovery',time_s=now,id=f'discovery:{h["id"]}',message=f'{h["scout_id"]}: YOLO + IMU defect reported',hazard_id=h['id']))
        decision=evaluate(routes,current['id'],progress,fixture['hazards'],now,last)
        if mode=='reroute' and decision['reroute']:
            choice=decision['reroute'];events.append(dict(type='reroute',id=f'reroute:{now}',time_s=now,
             message=f"Rerouting via Albert, Hickory and Hazel. {abs(choice['eta_delta_s']):.0f} seconds {'faster' if choice['eta_delta_s']<0 else 'extra' }.",
             from_route=current['id'],to_route=choice['route_id'],before=decision['current'],after=choice))
            current=next(r for r in routes if r['id']==choice['route_id']);progress=choice['progress_m'];last=now
            decision=evaluate(routes,current['id'],progress,fixture['hazards'],now,last)
        speed=lengths(current['coords'])[-1]/current['duration_s']
        for alert in warnings(current,progress,fixture['hazards'],now,warned,speed):
            warned.add(alert['hazard_id']);events.append(dict(**alert,id=f'warning:{alert["hazard_id"]}',time_s=now))
        arrived=progress>=lengths(current['coords'])[-1]
        states.append(dict(time_s=now,coords=position(current,progress),route_id=current['id'],progress_m=progress,
                           arrived=arrived,metrics=decision['current']))
        if arrived:break
        progress=min(lengths(current['coords'])[-1],progress+speed*.25);now=round(now+.25,2)
    return dict(states=states,events=events,duration_s=now)


def install_navigation_routes(app,here):
    fixture_path=here/'demo_data/navigation.json'
    @lru_cache(maxsize=1)
    def fixture():return json.loads(fixture_path.read_text())
    @lru_cache(maxsize=2)
    def replay(mode):return simulate(fixture(),mode)

    @app.post('/api/navigation/evaluate')
    def navigation(body:NavigationRequest):
        routes=[r.model_dump() for r in body.routes];hazards=[h.model_dump() for h in body.hazards]
        current=next(r for r in routes if r['id']==body.current_route_id)
        lateral,progress=project(body.location,current['coords'])
        if lateral>25:raise HTTPException(422,'Position is off the current route; request fresh route geometry first')
        result=evaluate(routes,current['id'],progress,hazards,body.now_s,body.last_reroute_s)
        result['warnings']=warnings(current,progress,hazards,body.now_s,set(body.warned_ids),body.speed_mps)
        result['progress_m']=progress
        return result

    @app.get('/api/navigation-demo')
    def demo(mode:Literal['warning','reroute']='reroute'):
        return dict(**fixture(),replay=replay(mode),mode=mode)

    @app.get('/api/navigation-audio/{clip}')
    def notification_audio(clip: str):
        if clip not in {'pothole_200m.mp3', 'rerouting.mp3'}:
            raise HTTPException(404, 'Unknown notification')
        path = here.parent / clip
        if not path.is_file():
            raise HTTPException(404, 'Notification audio unavailable')
        return FileResponse(path, media_type='audio/mpeg')

    @app.get('/navigation-demo')
    def page():return FileResponse(here/'static/navigation.html',headers={'Cache-Control':'no-store'})
