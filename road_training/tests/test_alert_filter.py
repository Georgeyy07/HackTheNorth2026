import math
import numpy as np
import pytest

from road_training.alert_filter import ScoreFilter, AlertPostprocessor


def row(i, p, *, final=True, valid=True):
    return dict(target_patch=i, is_final=final, valid=valid, probability=p,
                start_s=i*.16, end_s=(i+1)*.16, available_s=(i+3)*.16,
                target_sample_start=i*16, target_sample_end=(i+1)*16,
                emitted_after_samples=(i+3)*16, iri_m_per_km=2.7,
                event_transition='stale_source_event', event_id=99, disturbance=False)


def test_kalman_matches_hand_computed_update_and_stays_bounded():
    f = ScoreFilter(kind='kalman', q_over_r=.5)
    assert f.update(.2) == .2
    # P-=1+.5, K=1.5/2.5=.6, x=.2+.6*(.8-.2)=.56
    assert f.update(.8) == pytest.approx(.56)
    assert f.variance == pytest.approx(.6)
    for p in [0,1]*100:
        assert 0 <= f.update(p) <= 1
        assert f.variance > 0


def test_reset_and_no_future_influence():
    prefix = [.1,.3,.9,.7,.2]
    for space in ('probability','logit'):
        a=ScoreFilter(kind='kalman',space=space);b=ScoreFilter(kind='kalman',space=space)
        before=[a.update(p) for p in prefix]
        after=[b.update(p) for p in prefix + [1]*30]
        assert before == after[:len(prefix)]
        assert a.update(None) is None
        assert a.update(.8) == pytest.approx(.8)


def test_kalman_steady_gain_is_equivalent_to_matched_ema():
    alpha=.4;kalman=ScoreFilter(kind='kalman',q_over_r=alpha**2/(1-alpha))
    ema=ScoreFilter(kind='ema',alpha=alpha)
    for _ in range(100):kalman.update(.1);ema.update(.1)
    rng=np.random.default_rng(41)
    for p in rng.uniform(0,1,100):
        assert kalman.update(p) == pytest.approx(ema.update(p),abs=1e-12)


def test_provisional_revisions_do_not_mutate_filter_or_events():
    a=AlertPostprocessor(kind='ema',alpha=.5);b=AlertPostprocessor(kind='ema',alpha=.5)
    for i,p in enumerate([.2,.9,.8,.1]):
        preview=a.update(row(i,.99,final=False))
        assert preview['score_kind']=='unfiltered_provisional'
        assert a.last_target==i-1
        assert a.update(row(i,p)) == b.update(row(i,p))


def test_hysteresis_gap_censoring_and_no_repeated_commit():
    f=AlertPostprocessor(onset=.7,offset=.5)
    output=[f.update(row(i,p)) for i,p in enumerate([.69,.7,.55,.49,.9])]
    assert [r['disturbance'] for r in output] == [False,True,True,False,True]
    assert output[1]['event_transition']=='start' and output[1]['event_id']==1
    assert output[3]['event_transition']=='end' and output[3]['event_end_sample']==48
    assert output[4]['event_id']==2
    missing=f.update(row(5,None,valid=False))
    assert missing['event_transition']=='censored' and missing['event_id']==2
    assert missing['disturbance'] is None
    restarted=f.update(row(6,.9));assert restarted['event_id']==3
    assert restarted['iri_m_per_km']==2.7 and restarted['available_s']==1.44
    with pytest.raises(ValueError):f.update(row(6,.9))
