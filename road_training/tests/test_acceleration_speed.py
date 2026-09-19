import torch
import pytest

from road_training.acceleration_speed import select_channels, augment, CanonicalInputAdapter, AccelerationSpeedDataset
from road_training.dataset import RoadDataset
from road_training.instance_model import InstancePatchTST, InstanceRoadModel


def small_model():
    return InstanceRoadModel(InstancePatchTST(channels=4, d_model=8, n_heads=2,
        n_layers=1, ffn_dim=16, max_patches=4), statistics_mode='both').eval()


def test_both_predictions_are_independent_of_gyro_values_and_masks():
    torch.manual_seed(81)
    model = CanonicalInputAdapter(small_model())
    x = torch.randn(2, 64, 7, requires_grad=True)
    mask = torch.ones_like(x, dtype=torch.bool)
    reference = model(x, mask)
    changed = x.detach().clone(); changed[..., 3:6] = float('nan')
    changed_mask = mask.clone(); changed_mask[..., 3:6] = False
    actual = model(changed, changed_mask)
    for key in reference:
        assert torch.equal(reference[key], actual[key])
    (reference['roughness'].sum() + reference['disturbance_logit'].sum()).backward()
    assert torch.count_nonzero(x.grad[..., 3:6]) == 0
    assert torch.count_nonzero(x.grad[..., :3]) > 0


def test_rotation_preserves_acceleration_norm_speed_and_missing_speed():
    x = torch.randn(3, 64, 4); x[..., 2] += 9.81
    x[0, :, 3] = 0
    mask = torch.ones_like(x, dtype=torch.bool); mask[0, :, 3] = False
    original = x.clone()
    result = augment(x, mask, seed=52, step=13, rotate=True, noise=False)
    torch.testing.assert_close(result[..., :3].norm(dim=-1), x[..., :3].norm(dim=-1))
    assert torch.equal(result[..., 3], x[..., 3])
    assert torch.equal(original, x)
    noisy = augment(x, mask, seed=52, step=13, rotate=True, noise=True)
    assert not noisy[0, :, 3].any()
    assert torch.equal(noisy[..., 3], x[..., 3])


def test_core_model_accepts_only_four_channels_and_predicts_per_patch():
    model = small_model()
    x = torch.randn(2, 64, 7); mask = torch.ones_like(x, dtype=torch.bool)
    x4, m4 = select_channels(x, mask)
    torch.testing.assert_close(x4[..., 3], x[..., 6])
    for value in model(x4, m4).values():
        assert value.shape == (2, 4)
    with pytest.raises(ValueError):
        model(x, mask)


def test_record_projection_removes_gyro_only_support_for_indexing_helpers(monkeypatch):
    import numpy as np
    raw = np.arange(70, dtype=np.float32).reshape(10, 7)
    observed = np.ones((10, 7), dtype=bool)
    observed[2, [0, 1, 2, 6]] = False
    arrays = dict(x=raw, mask=observed)
    monkeypatch.setattr(RoadDataset, '_open', lambda self, index: arrays)
    dataset = object.__new__(AccelerationSpeedDataset)
    projected = dataset._open(0)
    assert projected['x'].shape == (10, 4)
    assert not projected['mask'][2].any()
    np.testing.assert_array_equal(projected['x'][:, 3], raw[:, 6])
    assert raw.shape == (10, 7) and observed[2].any()
    assert dataset._open(0)['x'] is projected['x']
