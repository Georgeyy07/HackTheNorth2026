"""Regression checks for loss extremes, patience and partially supervised updates."""
import copy

import pytest
import torch
from torch.nn import functional as F

from road_training.patchtst import PatchTST, PatchTSTRoadModel
from road_training.pretrain_rcd import update_patience
from road_training.train import focal_loss
from road_training.train_multitask import joint_loss, run_epoch


def binary_inputs(logits, labels):
    known = labels != -100
    output = dict(roughness=torch.zeros_like(logits), disturbance_logit=logits,
                  patch_valid=torch.ones_like(known))
    targets = dict(roughness=torch.zeros_like(logits), roughness_valid=torch.zeros_like(known),
                   disturbance=labels, disturbance_valid=known)
    return output, targets


@pytest.mark.parametrize('gamma', [.01, .1, .5, 0., 2.])
def test_focal_extremes_and_ignored_labels_have_finite_gradients(gamma):
    # Include exact/subnormal CE, very wrong predictions, and an unknown target.
    logits = torch.tensor([[-1000., 1000., -100., 100., 1000., -1000., 0.]], requires_grad=True)
    labels = torch.tensor([[0, 1, 0, 1, 0, 1, -100]])
    loss = joint_loss(*binary_inputs(logits, labels), gamma=gamma)['total']
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0, -1] == 0
    assert logits.grad[0, 4] > 0 and logits.grad[0, 5] < 0

    scores = torch.stack((torch.zeros_like(logits.detach()), logits.detach()), -1).requires_grad_()
    loss = focal_loss(scores, labels, gamma=gamma).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(scores.grad).all()
    assert (scores.grad[0, -1] == 0).all()


@pytest.mark.parametrize('gamma', [.1, .5, 2.])
def test_fractional_focal_matches_known_probabilities_and_gradients(gamma):
    logits = torch.tensor([[-2., -.3, .7, 2.]], dtype=torch.double, requires_grad=True)
    labels = torch.tensor([[0, 1, 0, 1]])
    # The reference is the direct probability definition, at nonsaturated inputs.
    signed = (2 * labels - 1) * logits
    pt = signed.sigmoid()
    reference = (-(1 - pt).pow(gamma) * pt.log()).mean()
    actual = joint_loss(*binary_inputs(logits, labels), gamma=gamma)['disturbance']
    torch.testing.assert_close(actual.double(), reference, rtol=1e-6, atol=1e-7)
    expected_gradient = torch.autograd.grad(reference, logits, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)


def test_default_gamma_two_preserves_previous_losses_and_gradients_exactly():
    torch.manual_seed(8)
    logits = torch.randn(3, 10, requires_grad=True)
    labels = torch.randint(2, (3, 10))
    ce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
    previous = ((-torch.expm1(-ce)).square() * ce).mean()
    current = joint_loss(*binary_inputs(logits, labels))['disturbance']
    torch.testing.assert_close(current, previous, rtol=0, atol=0)
    previous_grad = torch.autograd.grad(previous, logits, retain_graph=True)[0]
    current_grad = torch.autograd.grad(current, logits)[0]
    torch.testing.assert_close(current_grad, previous_grad, rtol=0, atol=0)


def test_patience_accumulates_small_improvements_and_still_stops_on_plateau():
    significant_best, stale = float('inf'), 0
    losses = [1 - i * .00004 for i in range(12)]
    for loss in losses:
        significant_best, stale = update_patience(loss, significant_best, stale)
        assert stale < 7  # Previously stopped at the eighth improving epoch.
    for loss in [losses[-1]] * 7:
        significant_best, stale = update_patience(loss, significant_best, stale)
    assert stale >= 7


def test_patience_reference_does_not_replace_the_best_checkpoint_value():
    significant_best, stale = update_patience(1., float('inf'), 0)
    best_checkpoint = 1.
    for loss in [.99996, .99992]:
        significant_best, stale = update_patience(loss, significant_best, stale)
        best_checkpoint = min(best_checkpoint, loss)
    assert significant_best == 1. and stale == 2
    assert best_checkpoint == .99992


def test_missing_task_does_not_update_a_head_with_existing_adam_momentum():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = PatchTSTRoadModel(PatchTST(patch_length=4, d_model=16, n_heads=2,
        n_layers=1, ffn_dim=32, dropout=0, max_patches=4), dropout=0)
    batch = dict(x=torch.randn(2, 16, 7), mask=torch.ones(2, 16, 7, dtype=torch.bool),
        labels=dict(localized_disturbance=torch.tensor([[0]*8 + [1]*8]*2),
            overall_iri=torch.full((2, 16), 2.), overall_iri_valid=torch.ones(2, 16, dtype=torch.bool),
            roughness_section=torch.zeros(2, 16, dtype=torch.long)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    for _ in range(2):
        run_epoch(model, [batch], 'cpu', optimizer, precision='fp32')
    for missing in ['roughness', 'disturbance']:
        masked = copy.deepcopy(batch)
        if missing == 'roughness':
            masked['labels']['overall_iri_valid'].fill_(False)
        else:
            masked['labels']['localized_disturbance'].fill_(-100)
        head = getattr(model, missing + '_head')
        before = {name: p.detach().clone() for name, p in head.named_parameters()}
        steps = {name: optimizer.state[p]['step'].clone() for name, p in head.named_parameters()}
        run_epoch(model, [masked], 'cpu', optimizer, precision='fp32')
        for name, p in head.named_parameters():
            assert p.grad is None
            assert torch.equal(p, before[name])
            assert torch.equal(optimizer.state[p]['step'], steps[name])
