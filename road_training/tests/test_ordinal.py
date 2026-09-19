import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from road_training.instance_model import InstancePatchTST, InstanceRoadModel
from road_training.ordinal import CUTPOINTS, OrdinalRoadModel, iri_classes, ordinal_loss, ordinal_probabilities
from road_training.ordinal_data import PVSDataset, pvs_patch_targets
from road_training.tools.prepare_pvs import import_record
from road_training.experiments.ordinal_study import section_results
from road_training.experiments.ordinal_report import QualityEnsemble, paired_block_interval


def checkpoint():
    encoder = InstancePatchTST(channels=4, d_model=8, n_heads=2, n_layers=1,
                              ffn_dim=16, max_patches=4, dropout=0)
    model = InstanceRoadModel(encoder, dropout=0).eval()
    return model, dict(config=dict(encoder_config=encoder.config,
                                   model_config=dict(statistics_mode='both', dropout=0)),
                       model_state=model.state_dict())


def test_iri_boundaries_unknown_values_and_exact_units():
    low, high = CUTPOINTS
    x = torch.tensor([0., low-1e-6, low, high, high+1e-6, -1., float('nan'), float('inf')], dtype=torch.float64)
    assert iri_classes(x).tolist() == [0, 0, 1, 1, 2, -100, -100, -100]
    assert np.allclose(np.array(CUTPOINTS)*63.36, [95., 170.])


def test_conversion_preserves_encoder_statistics_detector_and_initial_quality_score():
    torch.manual_seed(52)
    original, saved = checkpoint()
    ordinal = OrdinalRoadModel.from_regression(saved, mode='ordinal').eval()
    reg = OrdinalRoadModel.from_regression(saved, mode='regression').eval()
    x = torch.randn(2, 64, 4)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[0, :, 3] = False
    reference, new, control = original(x, mask), ordinal(x, mask), reg(x, mask)
    assert torch.equal(reference['disturbance_logit'], new['disturbance_logit'])
    assert torch.equal(reference['roughness'], new['quality_score'])
    assert torch.equal(new['quality_score'], control['roughness'])
    assert 'roughness' not in new  # An ordinal latent score is not reported as calibrated IRI.
    assert torch.equal(new['quality_class'], iri_classes(reference['roughness']))
    assert new['quality_logits'].shape == (2, 4, 2)
    assert new['quality_probability'].shape == (2, 4, 3)
    assert (new['quality_logits'][..., 0] >= new['quality_logits'][..., 1]).all()
    assert (new['quality_probability'] >= 0).all()
    torch.testing.assert_close(new['quality_probability'].sum(-1), torch.ones(2, 4))
    other_window = x.clone(); other_window[1] *= 100
    assert torch.equal(ordinal(other_window, mask)['quality_logits'][0], new['quality_logits'][0])


def test_cumulative_loss_known_only_weighting_and_gradients():
    logits = torch.tensor([[[2., -1.], [.5, -.5], [-2., -3.]]], requires_grad=True)
    labels = torch.tensor([[2, 1, -100]])
    valid = labels >= 0
    weights = torch.tensor([[2., 1., 9.]])
    expected = F.binary_cross_entropy_with_logits(logits[valid], torch.tensor([[1., 1.], [1., 0.]]), reduction='none').mean(-1)
    loss = ordinal_loss(logits, labels, valid, weights)
    torch.testing.assert_close(loss, (expected*torch.tensor([2., 1.])).sum()/3)
    loss.backward()
    assert not logits.grad[0, 2].any() and torch.isfinite(logits.grad).all()
    unknown = ordinal_loss(logits, labels, torch.zeros_like(valid))
    assert unknown == 0 and not unknown.requires_grad
    with pytest.raises(ValueError, match='0, 1 or 2'):
        ordinal_loss(logits, labels, torch.ones_like(valid))


def test_pvs_has_separate_ordered_thresholds_and_gradients_without_detector_labels():
    _, saved = checkpoint()
    model = OrdinalRoadModel.from_regression(saved, mode='ordinal')
    x = torch.randn(3, 64, 4)
    out = model(x, vehicle=torch.tensor([0, 1, 2]))
    assert (out['pvs_logits'][..., 0] > out['pvs_logits'][..., 1]).all()
    labels = torch.tensor([[0]*4, [1]*4, [2]*4])
    loss = ordinal_loss(out['pvs_logits'], labels, torch.ones_like(labels, dtype=torch.bool))
    loss.backward()
    assert model.pvs_start.grad is not None and model.pvs_gap.grad is not None
    assert model.log_temperature.grad is None  # Physical threshold temperature untouched by PVS loss.
    assert all(p.grad is None for p in model.disturbance_head.parameters())
    assert any(p.grad is not None for p in model.encoder.parameters())


def test_section_aggregation_equal_weights_traversals_and_absent_classes():
    rows = [dict(road='test', section=0, target=1., score=.5, probability=[.8,.15,.05]),
            dict(road='test', section=0, target=1., score=2., probability=[.4,.5,.1]),
            dict(road='test', section=1, target=2., score=2., probability=[.1,.8,.1])]
    result = section_results(rows, 'ordinal')
    assert result['sections'] == 2 and result['support'] == [1,1,0]
    assert result['rows'][0]['probability'] == pytest.approx([.6,.325,.075])
    assert result['macro_f1_present'] == 1.
    assert result['per_class']['bad']['recall'] is None
    assert result['macro_f1_all_three'] == pytest.approx(2/3)
    bad = copy.deepcopy(rows); bad[1]['target'] = 4.
    with pytest.raises(ValueError, match='inconsistent'):
        section_results(bad, 'ordinal')


def test_pvs_import_no_future_speed_correct_axes_stop_mask_and_window_boundaries(tmp_path):
    raw, output = tmp_path/'raw', tmp_path/'out'
    folder = raw/'PVS 1'; folder.mkdir(parents=True)
    output.mkdir()
    t = 1000.+np.arange(64)/100
    pd.DataFrame(dict(timestamp=t, acc_x_dashboard=np.ones(64), acc_y_dashboard=np.full(64, 2.),
                      acc_z_dashboard=np.full(64, 9.81))).to_csv(folder/'dataset_mpu_left.csv', index=False)
    pd.DataFrame(dict(timestamp=[1000.10,1000.30], speed_meters_per_second=[0., 10.])).to_csv(folder/'dataset_gps.csv', index=False)
    # CSV column order must not change the semantic class indices.
    pd.DataFrame(dict(bad_road_left=np.zeros(64, int), regular_road_left=np.zeros(64, int),
                      good_road_left=np.ones(64, int))).to_csv(folder/'dataset_labels.csv', index=False)
    record = import_record(raw, output, 1)
    x, mask, q, ready = [np.load(output/'pvs_1'/f'{name}.npy') for name in ('x','mask','quality','speed_source_time')]
    np.testing.assert_allclose(x[:, :3], np.tile([2., -1., 9.81], (64,1)), rtol=1e-6)
    assert not mask[:10, 3].any() and (x[:30, 3] == 0).all()
    assert (q[:30] == -100).all() and (q[30:] == 0).all()
    assert (ready[mask[:,3]] <= (t-t[0])[mask[:,3]]+1e-7).all()
    (output/'manifest.json').write_text(json.dumps(dict(sample_rate_hz=100,
        channels=['accel_x','accel_y','accel_z','speed'], records=[record])))
    data = PVSDataset(output, window_size=32, stride=1)
    assert len(data) == 33 and data[32]['x'].shape == (32,4)
    with pytest.raises(IndexError):
        data[33]
    batch = {k: data[32][k][None] for k in ('x','mask','quality')}
    target, valid = pvs_patch_targets(batch)
    assert target.tolist() == [[0,0]] and valid.all()
    batch['quality'][0, 0] = 1
    _, valid = pvs_patch_targets(batch)
    assert valid.tolist() == [[False,True]]


def test_ensemble_pools_probabilities_and_preserves_detector_across_quality_modes():
    torch.manual_seed(71)
    _, saved = checkpoint()
    second = copy.deepcopy(saved)
    # Deliberately distinct members expose accidental averaging of logits.
    second['model_state']['disturbance_head.4.bias'] += 4
    x = torch.randn(2, 64, 4)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[0, :16] = False
    regression = QualityEnsemble([OrdinalRoadModel.from_regression(s, mode='regression')
                                  for s in (saved, second)], 'regression').eval()
    ordinal = QualityEnsemble([OrdinalRoadModel.from_regression(s, mode='ordinal')
                               for s in (saved, second)], 'ordinal').eval()
    members = [m(x, mask) for m in ordinal.models]
    result, control = ordinal(x, mask), regression(x, mask)
    probability = torch.stack([o['quality_probability'] for o in members]).mean(0)
    detector = torch.stack([o['disturbance_logit'].sigmoid() for o in members]).mean(0)
    torch.testing.assert_close(result['quality_probability'], probability)
    torch.testing.assert_close(result['disturbance_logit'].sigmoid(), detector)
    assert torch.equal(result['disturbance_logit'], control['disturbance_logit'])
    assert not result['patch_valid'][0, 0]
    torch.testing.assert_close(probability.sum(-1), torch.ones(2, 4))
    assert (probability >= 0).all()
    with pytest.raises(ValueError, match='same quality mode'):
        QualityEnsemble(list(ordinal.models), 'regression')


def test_paired_spatial_bootstrap_rejects_unmatched_sections_and_identical_is_zero():
    rows = [dict(road='road', section=i, target=1.+i%2, score=1.+i%2,
                 probability=[.8,.15,.05] if i%2 == 0 else [.2,.7,.1]) for i in range(15)]
    result = dict(sections=section_results(rows, 'ordinal'))
    interval = paired_block_interval(result, result, repeats=100)
    assert interval['blocks'] == 3
    assert interval['delta_macro_f1'] == 0
    assert interval['percentile_95'] == [0., 0.]
    wrong = copy.deepcopy(result)
    wrong['sections']['rows'][0]['section'] = 999
    with pytest.raises(AssertionError):
        paired_block_interval(result, wrong, repeats=1)
