import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from route_planner.navigation import evaluate,warnings,lengths,position,project
from road_viewer.navigation import install_navigation_routes,simulate

HERE=Path(__file__).resolve().parents[2]/'road_viewer'
FIXTURE=json.loads((HERE/'demo_data/navigation.json').read_text())

def test_demo_has_causal_discoveries_and_one_continuous_reroute():
    f=FIXTURE;run=simulate(f,'reroute');reroutes=[e for e in run['events'] if e['type']=='reroute']
    assert len(reroutes)==1
    event=reroutes[0];assert event['time_s']==19
    assert event['after']['eta_delta_s']<0
    assert event['after']['potholes']<event['before']['potholes']
    assert all(s['route_id']=='recorded' for s in run['states'] if s['time_s']<19)
    assert run['states'][-1]['arrived']
    from alert_service.alert_math import haversine_distance_m
    assert max(haversine_distance_m(*a['coords'],*b['coords']) for a,b in zip(run['states'],run['states'][1:]))<5
    assert max(s['metrics']['potholes'] for s in run['states'] if s['time_s']<11.6)==0
    # A shared final hazard is still warned about after rerouting.
    assert len([e for e in run['events'] if e['type']=='warning'])==1


def test_warnings_use_forward_route_distance_and_do_not_repeat():
    run=simulate(FIXTURE,'warning');events=[e for e in run['events'] if e['type']=='warning']
    assert len(events)==3
    assert len({e['hazard_id'] for e in events})==3
    assert all(197<e['distance_m']<=200 for e in events)
    assert not any(e['type']=='reroute' for e in run['events'])
    route=FIXTURE['routes'][0];h=FIXTURE['hazards'][0];_,along=project(h['coords'],route['coords'])
    assert warnings(route,along+1,[h],300,set(),10)==[]
    assert warnings(route,along-100,[h],0,set(),10)==[]
    assert warnings(route,along-100,[h],300,set(),0)==[]
    assert warnings(route,along-100,[h],300,{h['id']},10)==[]


def test_reroute_rejects_cooldown_large_detour_and_disconnected_candidates():
    routes=FIXTURE['routes'];progress=150;h=FIXTURE['hazards']
    assert evaluate(routes,'recorded',progress,h,30)['reroute']
    assert not evaluate(routes,'recorded',progress,h,30,last_reroute_s=20)['reroute']
    expensive=[routes[0],dict(routes[1],duration_s=1000)]
    assert not evaluate(expensive,'recorded',progress,h,30)['reroute']
    distant=[routes[0],dict(routes[1],coords=[[a+.005,b] for a,b in routes[1]['coords']])]
    assert not evaluate(distant,'recorded',progress,h,30)['reroute']
    assert not evaluate(routes,'recorded',lengths(routes[0]['coords'])[-1]-50,h,300)['reroute']


def test_api_validates_and_evaluates_current_location():
    app=FastAPI();install_navigation_routes(app,HERE);client=TestClient(app)
    body=dict(routes=FIXTURE['routes'],current_route_id='recorded',location=position(FIXTURE['routes'][0],150),
              speed_mps=10,now_s=30,hazards=FIXTURE['hazards'])
    r=client.post('/api/navigation/evaluate',json=body);assert r.status_code==200,r.text
    assert r.json()['reroute']['route_id']=='bypass'
    body['location']=[0,0];assert client.post('/api/navigation/evaluate',json=body).status_code==422
    body['current_route_id']='missing';assert client.post('/api/navigation/evaluate',json=body).status_code==422
    assert client.get('/api/navigation-demo?mode=wrong').status_code==422
    assert client.get('/navigation-demo').status_code==200
