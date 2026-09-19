"""Check masked focal MixUp against explicit objectives and missing-task cases."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from road_training.mixup import mix_disturbance, mixed_joint_loss, latent_forward
from road_training.patchtst import PatchTST, PatchTSTRoadModel
from road_training.train_multitask import joint_loss
from road_training.experiments.real_mixup import sample


def targets():
    return dict(roughness=torch.tensor([[0., 0.], [0., 0.], [2., 3.]]),
        roughness_valid=torch.tensor([[False, False], [False, False], [True, True]]),
        disturbance=torch.tensor([[0, 1], [1, 0], [-100, -100]]),
        disturbance_valid=torch.tensor([[True, True], [True, True], [False, False]]))


def output():
    return dict(roughness=torch.tensor([[1., 2.], [3., 4.], [1., 5.]], requires_grad=True),
        disturbance_logit=torch.tensor([[-2., .7], [2., -.7], [0., 0.]], requires_grad=True),
        patch_valid=torch.ones(3, 2, dtype=torch.bool))


def recipe():
    return dict(partner=torch.tensor([1, 0, 2]), coefficient=torch.tensor([.25, .8, 1.]),
                active=torch.tensor([True, True, False]))


def test_zero_focal_exponent_matches_weighted_soft_bce():
    t, out, r = targets(), output(), recipe()
    alpha = torch.tensor([.4, 1.6])
    losses = mixed_joint_loss(out, t, r, gamma=0, alpha=alpha)
    y = t['disturbance'][:2].float()
    w = r['coefficient'][:2, None]
    soft = w*y + (1-w)*y.flip(0)
    logits = out['disturbance_logit'][:2]
    expected = (-alpha[1]*soft*F.logsigmoid(logits) - alpha[0]*(1-soft)*F.logsigmoid(-logits)).mean()
    torch.testing.assert_close(losses['disturbance'], expected)
    torch.testing.assert_close(losses['roughness'], F.smooth_l1_loss(out['roughness'][2], t['roughness'][2]))
    losses['total'].backward()
    assert torch.isfinite(out['disturbance_logit'].grad).all()
    assert not out['roughness'].grad[:2].any()


def test_focal_uses_expectation_of_two_hard_label_terms():
    t, out, r = targets(), output(), recipe()
    loss = mixed_joint_loss(out, t, r, gamma=2)
    a = F.binary_cross_entropy_with_logits(out['disturbance_logit'][:2], t['disturbance'][:2].float(), reduction='none')
    b = F.binary_cross_entropy_with_logits(out['disturbance_logit'][:2], t['disturbance'][:2].flip(0).float(), reduction='none')
    w = r['coefficient'][:2, None]
    torch.testing.assert_close(loss['disturbance'], (w*(1-a.neg().exp())**2*a + (1-w)*(1-b.neg().exp())**2*b).mean())


def test_unknown_partner_and_invalid_input_patch_never_supervise():
    t, out = targets(), output()
    t['disturbance'][1, 0] = -100
    t['disturbance_valid'][1, 0] = False
    out['patch_valid'][1, 1] = False
    losses = mixed_joint_loss(out, t, recipe(), alpha=torch.tensor([1., 2.]))
    assert losses['disturbance_valid'].tolist() == [[False, True], [False, False], [False, False]]
    losses['total'].backward()
    assert out['disturbance_logit'].grad[0, 0] == 0
    assert not out['disturbance_logit'].grad[1:].any()


def test_unmixed_objective_exactly_preserves_existing_loss():
    for r in (None, dict(recipe(), active=torch.zeros(3, dtype=torch.bool))):
        actual = mixed_joint_loss(output(), targets(), r, alpha=torch.tensor([1., 2.]))
        expected = joint_loss(output(), targets(), alpha=torch.tensor([1., 2.]))
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_never_mix_domains_missing_channels_or_roughness_examples():
    x = torch.arange(8*8*7, dtype=torch.float32).reshape(8, 8, 7)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[2:4, :, 6] = False  # RoadSens has no speed.
    mask[4:6, :, 3:6] = False  # LiRA has no gyro.
    mask[7, :, 6] = False  # Different Kaggle availability signature -> singleton.
    x[~mask] = 0
    t = dict(roughness=torch.zeros(8, 2), roughness_valid=torch.zeros(8, 2, dtype=torch.bool),
             disturbance=torch.zeros(8, 2, dtype=torch.long), disturbance_valid=torch.ones(8, 2, dtype=torch.bool))
    t['roughness_valid'][4:6] = True
    names = ['kaggle']*2+['roadsens']*2+['lira']*2+['kaggle']*2
    before = torch.get_rng_state()
    mixed, valid, r = mix_disturbance(x, mask, t, names, seed=42, step=0, probability=1)
    assert torch.equal(before, torch.get_rng_state())
    assert not r['active'][4:6].any() and not r['active'][7]
    assert torch.equal(mixed[4:6], x[4:6])
    assert not mixed[~valid].any()
    assert not valid[2:4, :, 6].any()
    for i, j in enumerate(r['partner'].tolist()):
        assert names[i] == names[j]
        assert torch.equal(mask[i].any(0), mask[j].any(0))
    again = mix_disturbance(x, mask, t, names, seed=42, step=0, probability=1)
    torch.testing.assert_close(mixed, again[0], rtol=0, atol=0)


def test_partial_missing_samples_use_intersection():
    x = torch.ones(3, 8, 7)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[1, 3, 0] = False
    x[1] *= 3
    x[~mask] = 0
    mixed, valid, r = mix_disturbance(x, mask, targets(), ['kaggle', 'kaggle', 'lira'], seed=4, step=3, probability=1)
    assert not valid[:2, 3, 0].any() and not mixed[:2, 3, 0].any()
    expected = r['coefficient'][0]+3*(1-r['coefficient'][0])
    torch.testing.assert_close(mixed[0, 0, 0], expected)


def test_sampler_preserves_source_slots_and_paired_draws():
    pools = dict(kaggle=np.arange(10), lira=np.arange(10, 20), roadsens=np.arange(20, 30))
    counts = dict(kaggle=64, lira=128, roadsens=64)
    a = np.array(sample(pools, counts, 42, 1, 4)).reshape(4, 256)
    assert np.array_equal(a.reshape(-1), sample(pools, counts, 42, 1, 4))
    assert (a[:, :64] < 10).all()
    assert ((a[:, 64:192] >= 10) & (a[:, 64:192] < 20)).all()
    assert (a[:, 192:] >= 20).all()


def test_mixed_roughness_is_rejected():
    r = recipe(); r['active'][2] = True
    with pytest.raises(ValueError, match='roughness'):
        mixed_joint_loss(output(), targets(), r)


def small_model():
    return PatchTSTRoadModel(PatchTST(patch_length=4, d_model=16, n_heads=2,
        n_layers=1, ffn_dim=32, max_patches=2, dropout=0), dropout=0).eval()


def test_latent_identity_equals_standard_model_and_iri_always_unchanged():
    model = small_model()
    x = torch.randn(3, 8, 7)
    mask = torch.ones_like(x, dtype=torch.bool)
    original = model(x, mask)
    identity = dict(partner=torch.arange(3), coefficient=torch.ones(3), active=torch.zeros(3, dtype=torch.bool))
    copied = latent_forward(model, x, mask, identity)
    for name in original:
        torch.testing.assert_close(copied[name], original[name], rtol=0, atol=0)
    mixed = latent_forward(model, x, mask, recipe())
    torch.testing.assert_close(mixed['roughness'], original['roughness'], rtol=0, atol=0)
    torch.testing.assert_close(mixed['disturbance_logit'][2], original['disturbance_logit'][2])


def test_latent_mixing_backpropagates_through_both_unmixed_waveforms():
    model = small_model()
    x = torch.randn(3, 8, 7, requires_grad=True)
    mask = torch.ones_like(x, dtype=torch.bool)
    observed = []
    hook = model.encoder.register_forward_pre_hook(lambda _, values: observed.append(values[0].detach().clone()))
    out = latent_forward(model, x, mask, recipe())
    hook.remove()
    torch.testing.assert_close(observed[0], x.detach(), rtol=0, atol=0)
    out['disturbance_logit'][0].sum().backward()
    assert x.grad[0].abs().sum() > 0 and x.grad[1].abs().sum() > 0
    assert not x.grad[2].any()


def test_latent_common_patch_mask_rejects_disjoint_missing_support():
    model = small_model()
    x = torch.randn(3, 8, 7)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[0, :4] = False
    out = latent_forward(model, x, mask, recipe())
    assert not out['patch_valid'][:2, 0].any()
    assert out['patch_valid'][:, 1].all()
