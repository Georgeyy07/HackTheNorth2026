import numpy as np
import pytest
from road_training.timeline import Timeline, filter_series


def push_scores(timeline,scores,iri=None,invalid=()):
    rows=[]
    for end in range(len(scores)+2):
        p=np.zeros(64);r=np.ones(64);v=np.zeros(64,bool)
        for age in range(3):
            target=end-age
            if 0<=target<len(scores):
                p[-1-age]=scores[target];r[-1-age]=iri[target] if iri is not None else 2.
                v[-1-age]=target not in invalid
        rows+=timeline.update(end,p,r,v)
    return rows


def test_same_patch_fusion_and_absolute_time_alignment():
    timeline=Timeline(defect_blend='recent',roughness_alpha=1.)
    rows=[]
    for end in range(6):
        p=np.zeros(64);r=np.ones(64);v=np.ones(64,bool)
        for age in range(3):p[-1-age]=.1*(end-age)+.02*age
        # Negative target placeholders cannot be valid probability payloads.
        p=np.maximum(p,0)
        rows+=timeline.update(end,p,r,v)
    assert [r['target_patch'] for r in rows]==[0,1,2,3]
    assert rows[0]['probability']==pytest.approx((.0+2*.02+3*.04)/6)
    assert all(r['emitted_after_samples']-r['target_sample_end']==32 for r in rows)
    assert all(r['votes']==3 for r in rows)
    assert timeline.finish()['uncommitted_patches']==[4,5]
    with pytest.raises(ValueError):timeline.update(5,p,r if isinstance(r,np.ndarray) else np.ones(64),v)


def test_hysteresis_events_and_stream_filter_equivalence():
    p=[.2,.7,.48,.46,.3,.7]
    timeline=Timeline(onset=.55,offset=.4,roughness_alpha=.5)
    rows=push_scores(timeline,p,iri=[1,2,3,4,5,6])
    labels,iri=filter_series(p,[1,2,3,4,5,6],[True]*6,onset=.55,offset=.4,roughness_alpha=.5)
    assert [r['disturbance'] for r in rows]==labels.tolist()==[False,True,True,True,False,True]
    np.testing.assert_allclose([r['iri_m_per_km'] for r in rows],iri)
    assert rows[1]['event_transition']=='start' and rows[1]['event_start_sample']==16
    assert rows[4]['event_transition']=='end' and rows[4]['event_end_sample']==64
    assert rows[5]['event_id']==2


def test_unknown_breaks_event_and_smoothing_without_becoming_normal():
    timeline=Timeline(roughness_alpha=.5)
    rows=push_scores(timeline,[.8]*4,iri=[1,1,10,10],invalid=(1,))
    assert rows[1]['valid'] is False and rows[1]['disturbance'] is None
    assert rows[1]['event_transition']=='censored'
    assert rows[2]['iri_m_per_km']==10 and rows[2]['event_id']==2
    timeline.reset();assert timeline.last_end==-1 and not timeline.pending


def test_no_revision_and_bounded_thirty_minute_session():
    timeline=Timeline()
    p=np.full(64,.7);r=np.ones(64);v=np.ones(64,bool)
    first=None
    for end in range(11250):  # 30 minutes at one 160-ms patch per update.
        rows=timeline.update(end,p if end<10 else 1-p,r,v)
        if end==2:first=dict(rows[0])
        assert len(timeline.pending)<=2
    assert first['disturbance'] is True and first['probability']==pytest.approx(.7)
    assert timeline.finish()['observed_samples']==180000
