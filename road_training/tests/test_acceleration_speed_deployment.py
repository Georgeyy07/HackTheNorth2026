"""Contract checks for the four-input model and finalized-score filtering."""
import copy
import json

import numpy as np
import pytest
import torch

from road_training.alert_filter import AlertPostprocessor, DEFAULT_ALERT_FILTER
from road_training.checkpoints import load_teachers, make_model
from road_training.common import sha
from road_training.export_test_drives import infer_drive
from road_training.timeline_stream import DEFAULT_CONFIG, RoadTimelineStream


class FourInputModel(torch.nn.Module):
    channels = 4

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, mask):
        assert x.shape[-1] == 4
        patches = x.masked_fill(~mask, 0).reshape(1, 64, 16, 4)
        # Speed changes the score, so projecting gyro_x as speed fails tests.
        value = patches[..., 0].mean(2) + patches[..., 3].mean(2)
        return dict(roughness=value.abs(), disturbance_logit=value,
                    patch_valid=mask.reshape(1, 64, 16, 4).any((2, 3)))


def test_live_filter_matches_separate_postprocessing_and_resets_all_state():
    torch.manual_seed(53)
    x = torch.randn(16 * 12 + 7, 4)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[48:64] = False
    plain_config = {k: v for k, v in DEFAULT_CONFIG.items() if k != 'alert_filter'}
    plain = RoadTimelineStream(FourInputModel(), plain_config).push(x, mask)
    filter_ = AlertPostprocessor(**DEFAULT_ALERT_FILTER)
    expected = [filter_.update(row) for row in plain]
    live = RoadTimelineStream(FourInputModel())
    actual = []
    for lo, hi in [(0, 7), (7, 69), (69, 103), (103, len(x))]:
        actual += live.push(x[lo:hi], mask[lo:hi])
        state = (live.alerts.last_target, live.alerts.filter.mean, live.alerts.filter.variance)
        for _ in range(4):
            assert all(row['score_kind'] == 'unfiltered_provisional' for row in live.provisional())
        assert state == (live.alerts.last_target, live.alerts.filter.mean, live.alerts.filter.variance)
    assert actual == expected
    assert all(a['iri_m_per_km'] == b['iri_m_per_km'] for a, b in zip(actual, plain))
    assert actual[3]['probability'] is None
    assert actual[4]['probability'] == actual[4]['original_probability']  # Reset after a gap.
    assert live.window.shape == (1, 1024, 4)
    assert live.finish()['incomplete_samples'] == 7
    live.reset()
    assert live.push(x, mask) == actual
    with pytest.raises(ValueError, match=r'samples,4'):
        live.push(torch.zeros(16, 7))


def test_finish_uses_filtered_event_state_and_never_finalizes_unseen_tail():
    # Raw consensus exceeds .6; the filtered deployment onset is .7.
    x = torch.full((80, 4), 0.)
    x[:, 0] = torch.logit(torch.tensor(.65))
    stream = RoadTimelineStream(FourInputModel())
    rows = stream.push(x)
    assert stream.timeline.active
    assert not any(row['disturbance'] for row in rows)
    finish = stream.finish()
    assert finish['active_event_id'] is None and not finish['event_end_censored']
    assert finish['uncommitted_patches'] == [3, 4]


def test_export_discards_gyro_before_inference_and_preserves_partial_tail():
    rng = np.random.default_rng(55)
    x = rng.normal(size=(16 * 10 + 7, 7)).astype(np.float32)
    mask = np.ones_like(x, bool)
    # A gyro-only patch must have no valid prediction after projection.
    mask[32:48, [0, 1, 2, 6]] = False
    expected, finish = infer_drive(FourInputModel(), x[:, [0, 1, 2, 6]],
                                  mask[:, [0, 1, 2, 6]], DEFAULT_CONFIG)
    changed = x.copy()
    changed[:, 3:6] = np.nan
    mask[:, 3:6] = False
    updates = []
    actual, actual_finish = infer_drive(FourInputModel(), changed, mask, DEFAULT_CONFIG, updates.append)
    assert expected == actual and finish == actual_finish
    assert actual[2]['probability'] is None
    assert actual[-1]['target_sample_end'] == len(x)
    assert actual[-1]['status'] == 'provisional_partial'
    assert actual[-1]['probability'] == actual[-1]['original_probability']
    assert len([r for r in updates if r['is_final']]) == 8
    assert all(r['score_kind'] == 'unfiltered_provisional' for r in actual[-3:])
    altered = changed.copy()
    altered[96:, 0] += 100
    future, _ = infer_drive(FourInputModel(), altered, mask, DEFAULT_CONFIG)
    assert [r for r in actual if r['emitted_after_samples'] <= 96] == [
        r for r in future if r['emitted_after_samples'] <= 96]
    changed[-1, 0] = np.nan
    with pytest.raises(ValueError, match='Observed inputs'):
        infer_drive(FourInputModel(), changed, mask, DEFAULT_CONFIG)


def test_checkpoint_receipt_checks_channels_order_and_member_geometry(tmp_path):
    config = dict(encoder_config=dict(channels=4, d_model=8, n_heads=2, n_layers=1,
                                      ffn_dim=16, max_patches=4, dropout=0),
                  model_config=dict(statistics_mode='both'),
                  input_channels=['accel_x', 'accel_y', 'accel_z', 'speed'])
    model = make_model(config).eval()
    weights = tmp_path / 'four.pt'
    torch.save(dict(config=config, model_state=model.state_dict()), weights)
    receipt = tmp_path / 'ensemble.json'
    contents = dict(checkpoints={'four.pt': sha(weights)}, input_channels=config['input_channels'])
    receipt.write_text(json.dumps(contents))
    loaded = load_teachers(receipt, device='cpu')
    x = torch.randn(1, 64, 4)
    with torch.no_grad():
        before, after = model(x), loaded(x, torch.ones_like(x, dtype=torch.bool))
    for key in before:
        torch.testing.assert_close(before[key], after[key])
    contents['input_channels'] = ['speed', 'accel_x', 'accel_y', 'accel_z']
    receipt.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match='channels/order'):
        load_teachers(receipt, device='cpu')
    other = copy.deepcopy(config)
    other['encoder_config']['channels'] = 7
    other.pop('input_channels')
    seven = make_model(other)
    torch.save(dict(config=other, model_state=seven.state_dict()), tmp_path / 'seven.pt')
    receipt.write_text(json.dumps(dict(checkpoints={'four.pt': sha(weights), 'seven.pt': sha(tmp_path / 'seven.pt')})))
    with pytest.raises(ValueError, match='same input channels'):
        load_teachers(receipt, device='cpu')


def test_default_file_and_python_config_match():
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / 'configs/timeline.json'
    assert json.loads(path.read_text())['config'] == DEFAULT_CONFIG


def test_fresh_training_plan_uses_four_inputs_and_scoped_augmentation(monkeypatch, tmp_path):
    from road_training.acceleration_speed import AccelerationSpeedDataset, augment
    from road_training.experiments import instance_study as base, mounting_study as mounting
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'manifest.json').write_text(json.dumps(dict(roadsens_alignment={}, records=[])))
    monkeypatch.setattr(base, 'DATA', data)
    monkeypatch.setattr(base, 'OUT', tmp_path / 'parent')
    base.initialize()
    plan = mounting.initialize(tmp_path / 'four', data_root=data, seed=52, epochs=1, steps_per_epoch=1)
    assert plan['channels'] == 4
    assert plan['input_channels'] == ['accel_x', 'accel_y', 'accel_z', 'speed']
    assert plan['arms'][mounting.ARM]['statistics_mode'] == 'both'
    assert base.dataset_class(plan['channels']) is AccelerationSpeedDataset
    previous = base.perturb
    x = torch.randn(2, 64, 4)
    x[..., 2] += 9.81
    mask = torch.ones_like(x, dtype=torch.bool)
    with mounting.configured_study(tmp_path / 'four'):
        actual = base.perturb(x, mask, seed=52, step=0)
        expected = augment(x, mask, seed=52, step=0, **plan['mounting_augmentation'])
        assert torch.equal(actual, expected)
        assert torch.equal(actual[..., 3], x[..., 3])
    assert base.perturb is previous
