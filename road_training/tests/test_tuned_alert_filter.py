import pytest
from road_training.tuned_alert_filter import AdaptiveScoreFilter, noise_ratio, make_processor


def test_adaptive_filter_uses_larger_q_only_for_current_shock():
    f=AdaptiveScoreFilter(gain=.5,jump=.2,fast_gain=.97)
    assert f.update(.1)==.1
    q=noise_ratio(.97);k=(1+q)/(2+q)
    assert f.update(.9)==pytest.approx(.1+k*.8)
    assert f.q==noise_ratio(.5)
    variance=f.variance;old=f.mean;p=old-.01;k=(variance+.5)/(variance+1.5)
    assert f.update(p)==pytest.approx(old+k*(p-old))


def test_light_filter_and_adaptive_filter_are_causal_and_reset_on_missing():
    for config in [dict(gain=.98),dict(gain=.5,jump=.2,fast_gain=.97),dict(gain=.9,space='logit')]:
        a=AdaptiveScoreFilter(**config);b=AdaptiveScoreFilter(**config)
        prefix=[.1,.2,.7,.8,.4,.1]
        left=[a.update(x) for x in prefix]
        right=[b.update(x) for x in prefix+[1,1,1]]
        assert left==right[:len(prefix)]
        assert a.update(None) is None
        assert a.update(.7)==pytest.approx(.7)


def test_bad_adaptive_parameters_rejected():
    with pytest.raises(ValueError):AdaptiveScoreFilter(gain=.9,jump=.2,fast_gain=.5)
    with pytest.raises(ValueError):AdaptiveScoreFilter(gain=1.)
    with pytest.raises(ValueError):AdaptiveScoreFilter(jump=0.)
