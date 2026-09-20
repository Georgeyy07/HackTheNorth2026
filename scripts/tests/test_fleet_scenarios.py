from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from scripts.generate_fleet_scenarios import stationary_intervals, yolo_for_patch, select_start, materialize


def test_stops_require_observed_contiguous_speed():
    times=np.arange(700)/100;speed=np.zeros(700);speed[200]=np.nan;speed[450:]=1
    frame=pd.DataFrame(dict(time_s=times,speed=speed))
    assert stationary_intervals(frame,.3,2)==[(0.,2.),(2.01,4.5)]
    # A clock gap splits what otherwise looks like a single stop.
    frame=pd.DataFrame(dict(time_s=np.r_[np.arange(100)/100,3+np.arange(100)/100],speed=0))
    assert stationary_intervals(frame,.3,2)==[]


def test_yolo_unknown_is_distinct_from_negative_and_patch_boundaries():
    assert yolo_for_patch(None,0,.16) is None
    vision=([.40,.433,.466,.50,.533,.566],[False,False,True,False,False,False],1/30)
    assert yolo_for_patch(vision,0,.16) is None
    assert yolo_for_patch(vision,.4,.56) is True
    assert yolo_for_patch(vision,.5,.58) is False
    assert yolo_for_patch(vision,.5,.8) is None
    assert yolo_for_patch(([.16],[True],1/30),0,.16) is None


def test_start_is_within_stop_with_two_seconds_remaining():
    samples=pd.DataFrame(dict(time_s=np.arange(600)/100,speed=np.r_[np.ones(100),np.zeros(500)]))
    rows=[dict(_source_start_s=.96,_source_end_s=1.12),dict(_source_start_s=1.12,_source_end_s=1.28)]
    row,proof=select_start(rows,[(1.,6.)],samples,2)
    assert row is rows[1] and proof['speed_mps']==0 and proof['stationary_remaining_s']>=2
    with pytest.raises(ValueError):select_start(rows,[(1.,2.)],samples,2)


def test_cascade_has_identical_observations_shifted_exactly_20_seconds():
    route=[dict(_source_us=t,latitude=43.,longitude=-80.,imu_defect_detected=True,
                yolo_pothole_detected=None,road_quality='bad') for t in (1000000,1160000,7000000)]
    starts=[dict(source_us=1000000),dict(source_us=7000000)]
    rows,plans=materialize(route,starts,'cascade','test',0,3,20)
    cars=[[r for r in rows if r['carID']==p['carID']] for p in plans]
    for i,car in enumerate(cars):
        assert len(car)==3
        for a,b in zip(cars[0],car):
            assert (datetime.fromisoformat(b['timestamp'])-datetime.fromisoformat(a['timestamp'])).total_seconds()==i*20
            assert {k:v for k,v in a.items() if k not in ('carID','timestamp')}=={k:v for k,v in b.items() if k not in ('carID','timestamp')}
    # The six-second source gap is retained, not compressed or interpolated.
    assert (datetime.fromisoformat(cars[0][-1]['timestamp'])-datetime.fromisoformat(cars[0][0]['timestamp'])).total_seconds()==6
    staggered,plans=materialize(route,starts,'staggered','test',0,99,20)
    assert len(plans)==2
    assert plans[0]['launch_timestamp']==plans[1]['launch_timestamp']
    assert plans[1]['rows']==1
