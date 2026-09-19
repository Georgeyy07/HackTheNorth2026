"""Physical geometry, masked sensors and trainer integration for mount errors."""
import math

import pytest
import torch

from road_training.mounting_augmentation import axis_angle, mounting_rotations, perturb


def inputs(batch=16, steps=24, up=2, device='cpu'):
    generator = torch.Generator(device=device).manual_seed(117)
    x = torch.randn(batch, steps, 7, device=device, generator=generator) * .2
    x[..., up] += 9.81
    x[..., 6] = 12.
    return x, torch.ones_like(x, dtype=torch.bool)


def test_known_yaw_rotation_sign_and_right_handed_axes():
    rotation = axis_angle(torch.tensor([[0., 0., 1.]]), torch.tensor([math.pi/2]))
    forward = torch.tensor([[[1., 0., 0.]]])
    torch.testing.assert_close(forward @ rotation.transpose(-1, -2),
                               torch.tensor([[[0., 1., 0.]]]), rtol=0, atol=1e-6)


@pytest.mark.parametrize('up', [1, 2])
def test_yaw_about_gravity_preserves_vertical_for_each_dataset_frame(up):
    x, mask = inputs(batch=2048, steps=2, up=up)
    x[..., :3] = 0; x[..., up] = 9.81
    matrix = mounting_rotations(x, mask, seed=4, step=0, tilt_degrees=0., probability=1.)
    rotated = x[..., :3] @ matrix.transpose(-1, -2)
    torch.testing.assert_close(rotated, x[..., :3], atol=2e-6, rtol=0)
    angle = ((matrix.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1).acos()
    assert angle.max() <= math.radians(20.) + 1e-5
    assert angle.max() >= math.radians(19.)  # The intended range is exercised.
    horizontal = (up+1) % 3
    assert not torch.allclose(matrix[:, horizontal, horizontal], torch.ones(len(x)))


@pytest.mark.parametrize('up', [1, 2])
def test_proper_rotation_and_tilt_bounds_without_changing_magnitude(up):
    x, mask = inputs(batch=2048, steps=4, up=up)
    matrix = mounting_rotations(x, mask, seed=9, step=4, probability=1.)
    eye = torch.eye(3).expand_as(matrix)
    torch.testing.assert_close(matrix @ matrix.transpose(-1, -2), eye, atol=5e-7, rtol=0)
    torch.testing.assert_close(torch.linalg.det(matrix), torch.ones(len(x)), atol=5e-7, rtol=0)
    vertical = torch.nn.functional.normalize(x[..., :3].mean(1), dim=-1)
    rotated = (matrix @ vertical[..., None]).squeeze(-1)
    tilt = (vertical*rotated).sum(-1).clamp(-1, 1).acos()
    assert tilt.max() <= math.radians(10.) + 1e-5
    assert tilt.max() >= math.radians(9.)


def test_same_rotation_through_time_for_both_triads_and_speed_unchanged():
    x, mask = inputs()
    x[..., 3:6] = x[..., :3] * .01
    before = x.clone()
    result = perturb(x, mask, seed=4, step=8, probability=1., noise=False)
    matrix = mounting_rotations(x, mask, seed=4, step=8, probability=1.)
    torch.testing.assert_close(result[..., :3], x[..., :3] @ matrix.transpose(-1, -2))
    torch.testing.assert_close(result[..., 3:6], result[..., :3] * .01)
    for offset in (0, 3):
        torch.testing.assert_close(result[..., offset:offset+3].norm(dim=-1),
                                   x[..., offset:offset+3].norm(dim=-1), atol=2e-6, rtol=2e-7)
    assert torch.equal(result[..., 6], x[..., 6])
    assert torch.equal(x, before)  # Does not mutate the dataset's raw window.


def test_deterministic_seed_and_step_without_touching_global_rng():
    x, mask = inputs()
    rng = torch.get_rng_state().clone()
    first = perturb(x, mask, seed=8, step=3)
    assert torch.equal(first, perturb(x, mask, seed=8, step=3))
    assert not torch.equal(first, perturb(x, mask, seed=8, step=4))
    assert not torch.equal(first, perturb(x, mask, seed=9, step=3))
    assert torch.equal(rng, torch.get_rng_state())


def test_clean_fraction_and_disabling_rotation():
    x, mask = inputs(batch=4096, steps=2)
    matrix = mounting_rotations(x, mask, seed=2, step=7)
    unchanged = (matrix == torch.eye(3)).all(-1).all(-1).float().mean()
    assert .22 <= unchanged <= .28
    for options in (dict(probability=0.), dict(rotate=False), dict(yaw_degrees=0., tilt_degrees=0.)):
        assert torch.equal(perturb(x, mask, seed=2, step=7, noise=False, **options), x)


def test_missing_gyro_padding_and_missing_speed_remain_missing():
    x, mask = inputs(batch=3)
    mask[0, :, 3:6] = False  # LiRA has no gyro.
    mask[1, :, 6] = False  # RoadSens has no speed.
    mask[:, :4, :] = False  # Beginning-of-drive padding.
    x[~mask] = float('nan')
    before_mask = mask.clone()
    result = perturb(x, mask, seed=5, step=1, probability=1.)
    assert torch.isfinite(result).all() and not result[~mask].any()
    assert torch.equal(mask, before_mask)
    assert torch.equal(result[..., 6][mask[..., 6]], x[..., 6][mask[..., 6]])


def test_partial_triads_and_missing_gravity_skip_whole_window_rotation():
    x, mask = inputs(batch=4)
    mask[0, 5, 0] = False
    mask[1, 8, 4] = False
    mask[2, :, :3] = False
    x[3, :, :3] = 0
    matrix = mounting_rotations(x, mask, seed=2, step=0, probability=1.)
    assert torch.equal(matrix, torch.eye(3).expand_as(matrix))
    result = perturb(x, mask, seed=2, step=0, probability=1., noise=False)
    assert torch.equal(result, x.masked_fill(~mask, 0))


def test_held_sensor_readings_stay_identical_with_noise():
    x, mask = inputs(steps=12)
    x = x.repeat_interleave(2, dim=1); mask = mask.repeat_interleave(2, dim=1)
    result = perturb(x, mask, seed=12, step=9, probability=1., noise=True)
    assert torch.equal(result[:, ::2], result[:, 1::2])


def test_other_windows_do_not_influence_a_windows_rotation():
    x, mask = inputs()
    before = mounting_rotations(x, mask, seed=4, step=2)
    x[1:] *= -2
    after = mounting_rotations(x, mask, seed=4, step=2)
    assert torch.equal(before[0], after[0])


@pytest.mark.parametrize('options', [dict(yaw_degrees=-1.), dict(yaw_degrees=float('nan')),
    dict(tilt_degrees=91.), dict(probability=1.1)])
def test_invalid_configuration_rejected(options):
    x, mask = inputs()
    with pytest.raises(ValueError):
        mounting_rotations(x, mask, seed=2, step=1, **options)


def test_nonfinite_observed_value_rejected_but_missing_payload_is_ignored():
    x, mask = inputs()
    x[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        mounting_rotations(x, mask, seed=0, step=0)
    mask[0, 0, 0] = False
    assert torch.isfinite(perturb(x, mask, seed=0, step=0)).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_autocast_preserves_fp32_rotation_and_finite_input_gradient(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    x, mask = inputs(device=device); x.requires_grad_()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        result = perturb(x, mask, seed=6, step=0, probability=1., noise=False)
    expected = perturb(x, mask, seed=6, step=0, probability=1., noise=False)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_trainer_adapter_is_scoped_and_cannot_augment_validation(monkeypatch, tmp_path):
    from road_training.experiments import mounting_study, instance_study
    previous_out, previous_perturb = instance_study.OUT, instance_study.perturb
    previous_data = instance_study.DATA
    options = dict(yaw_degrees=18., tilt_degrees=8., probability=1.)
    monkeypatch.setattr(instance_study, 'check_plan', lambda: dict(mounting_augmentation=options))
    monkeypatch.setattr(mounting_study, 'read', lambda path: dict(data_root=str(tmp_path/'aligned')))
    original_eval = instance_study.run_epoch
    with pytest.raises(RuntimeError, match='training stopped'):
        with mounting_study.configured_study(tmp_path):
            assert instance_study.OUT == tmp_path.resolve()
            assert instance_study.DATA == tmp_path/'aligned'
            assert instance_study.run_epoch is original_eval
            x, mask = inputs()
            actual = instance_study.perturb(x, mask, seed=5, step=3, rotate=True, noise=True)
            expected = perturb(x, mask, seed=5, step=3, **options)
            assert torch.equal(actual, expected)
            raise RuntimeError('training stopped')
    assert instance_study.OUT == previous_out
    assert instance_study.DATA == previous_data
    assert instance_study.perturb is previous_perturb
