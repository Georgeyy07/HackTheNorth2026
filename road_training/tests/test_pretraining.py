"""Paper invariants: target isolation, retained positions, diffusion algebra."""
import math

import pytest
import torch

from road_training.arctan import ArcTanEncoder, ArcTanPretrainer, DiTBlock, arctan_loss
from road_training.droppatch import DropPatchPretrainer
from road_training.patchtst import PatchTST, PatchTSTRoadModel
from road_training.pretrain import build_pretrainer, load_pretrained_encoder, run_epoch


@pytest.fixture(autouse=True)
def small_cpu_threads():
    torch.set_num_threads(1)
    torch.manual_seed(9)


def config():
    return dict(channels=3, patch_length=4, d_model=16, n_heads=2,
                n_layers=2, ffn_dim=32, dropout=0., max_patches=64)


def test_droppatch_paper_counts_and_actual_attention_length():
    model = DropPatchPretrainer(PatchTST(**config()))
    lengths = []
    handle = model.encoder.layers[0].register_forward_pre_hook(lambda _, args: lengths.append(args[0].shape[1]))
    result = model(torch.randn(2, 256, 3), generator=torch.Generator().manual_seed(1))
    handle.remove()
    assert lengths == [25]  # floor(64 * .4); dropped tokens never enter attention.
    assert result["prediction"].shape == (2, 3, 25, 4)
    assert (result["patch_mask"].sum(-1) == 10).all()
    assert (result["kept_indices"].sort(-1).values.diff() > 0).all()
    expected = (result["prediction"] - result["target"])[result["loss_mask"]].square().mean()
    torch.testing.assert_close(result["loss"], expected)


def test_droppatch_preserves_original_positions_before_attention():
    encoder = PatchTST(**config()).eval()
    with torch.no_grad():
        encoder.value_embedding.weight.zero_()
        encoder.value_embedding.bias.zero_()
        encoder.valid_embedding.weight.zero_()
        encoder.position.copy_(torch.arange(64).view(1, 1, 64, 1).expand_as(encoder.position))
    kept = torch.tensor([[[7, 0, 4], [2, 6, 1], [5, 3, 0]]])
    seen = []
    handle = encoder.layers[0].register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
    DropPatchPretrainer(encoder)(torch.randn(1, 32, 3), kept_indices=kept)
    handle.remove()
    torch.testing.assert_close(seen[0][..., 0], kept.reshape(3, 3).float())


def test_dropped_and_masked_targets_do_not_leak_into_predictions_or_input_gradient():
    model = DropPatchPretrainer(PatchTST(**config())).eval()
    x = torch.randn(2, 32, 3)
    kept = torch.tensor([0, 3, 6]).view(1, 1, 3).expand(2, 3, 3)
    hidden = torch.tensor([False, True, False]).view(1, 1, 3).expand_as(kept)
    context = torch.zeros_like(x, dtype=torch.bool)
    context[:, :4] = True
    context[:, 24:28] = True
    altered = x.masked_fill(~context, 1234)
    before = model(x, kept_indices=kept, patch_mask=hidden)
    after = model(altered, kept_indices=kept, patch_mask=hidden)
    torch.testing.assert_close(before["prediction"], after["prediction"], atol=0, rtol=0)
    assert not torch.equal(before["target"], after["target"])
    x.requires_grad_()
    model(x, kept_indices=kept, patch_mask=hidden)["loss"].backward()
    assert (x.grad[~context] == 0).all()
    assert x.grad[context].abs().sum() > 0


@pytest.mark.parametrize("method", ["droppatch", "arctan"])
def test_missing_payloads_and_channels_never_become_targets(method):
    model = build_pretrainer(method, config()).eval()
    x = torch.randn(2, 32, 3)
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[..., 2] = False
    valid[:, 12:15, 0] = False
    def run(payload):
        return model(x.masked_fill(~valid, payload), valid, generator=torch.Generator().manual_seed(11))
    a, b = run(float("nan")), run(999.)
    torch.testing.assert_close(a["prediction"], b["prediction"], atol=0, rtol=0)
    torch.testing.assert_close(a["loss"], b["loss"], atol=0, rtol=0)
    assert not a["loss_mask"][:, 2].any()
    assert torch.isfinite(a["prediction"]).all()
    all_missing = model(x, torch.zeros_like(valid))
    assert all_missing["loss"].item() == 0
    assert torch.isfinite(all_missing["prediction"]).all()
    all_missing["loss"].backward()


def test_droppatch_sparse_channels_keep_context_and_pad_without_loss():
    model = DropPatchPretrainer(PatchTST(**config()), drop_ratio=.99)
    x = torch.randn(1, 32, 3)
    valid = torch.zeros_like(x, dtype=torch.bool)
    valid[:, :8, 0] = True  # Two patches: retain both, hide one.
    valid[:, :4, 1] = True  # One patch: context only, no reconstruction.
    result = model(x, valid)
    assert result["patch_mask"].sum().item() == 1
    assert result["loss_mask"].sum().item() == 4
    assert result["retained_mask"].sum().item() == 3


def test_arctan_loss_matches_scalar_formula_and_analytic_gradient():
    prediction = torch.tensor([.7, -.2, 3.], requires_grad=True)
    target = torch.tensor([1.2, 3., 2.9])
    values = arctan_loss(target, prediction)
    expected = []
    for y, p in zip(target.tolist(), prediction.tolist()):
        e = y - p
        expected.append(.5 * (-1.1 / 2.6 * math.log1p(1.3**2 * e**2)
                              + 1.1 * e * math.atan(1.3 * e)) + .5 * abs(e) + .05 * p**2)
    torch.testing.assert_close(values, torch.tensor(expected))
    values.sum().backward()
    error = target - prediction.detach()
    derivative = -.55 * torch.atan(1.3 * error) - .5 * error.sign() + .1 * prediction.detach()
    torch.testing.assert_close(prediction.grad, derivative)
    # The gamma term is on prediction, so zero residual != zero loss.
    torch.testing.assert_close(arctan_loss(torch.tensor(2.), torch.tensor(2.)), torch.tensor(.2))


@pytest.mark.parametrize("channel_mode", ["mixed", "independent"])
def test_arctan_corruption_velocity_clamp_and_patch_reconstruction(channel_mode):
    encoder = ArcTanEncoder(**config(), channel_mode=channel_mode)
    model = ArcTanPretrainer(encoder).eval()
    x = torch.arange(24).reshape(1, 8, 3).float() / 10
    patches, _ = encoder.patchify(x)
    noise = torch.full_like(patches, .5)
    for t_value in (0., .5, .99, 1.):
        times = torch.full((1, encoder.feature_channels, 2), t_value)
        result = model(x, timesteps=times, noise=noise)
        expected_noisy = t_value * patches + (1 - t_value) * noise
        torch.testing.assert_close(result["noisy"], expected_noisy)
        torch.testing.assert_close(result["target_velocity"],
                                   (patches - expected_noisy) / max(1 - t_value, .05))
        torch.testing.assert_close(result["predicted_velocity"],
                                   (result["prediction"].float() - expected_noisy) / max(1 - t_value, .05))
        assert result["prediction"].shape == (1, 3, 2, 4)
        assert torch.isfinite(result["loss"])
        result["loss"].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)
    # pack/unpack retain sample ordering and channel membership.
    torch.testing.assert_close(encoder.unpack(encoder.pack(patches), 1), patches)


def test_adaln_zero_starts_as_identity_but_gates_and_condition_can_learn():
    block = DiTBlock(16, 2, 32, 0.)
    x, condition = torch.randn(2, 8, 16), torch.randn(2, 8, 16)
    valid = torch.ones(2, 8, dtype=torch.bool)
    out = block(x, condition, valid)
    torch.testing.assert_close(out, x, atol=0, rtol=0)
    out.square().sum().backward()
    assert block.modulation[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        block.modulation[-1].weight.normal_(std=.1)
    assert not torch.allclose(block(x, condition, valid), block(x, condition + 1, valid))


@pytest.mark.parametrize("channel_mode", ["mixed", "independent"])
def test_clean_endpoint_transfers_projection_and_two_heads(channel_mode, tmp_path):
    encoder_config = {**config(), "channel_mode": channel_mode}
    stats = dict(mean=[1, 2, 3], std=[2, 3, 4])
    pretrainer = build_pretrainer("arctan", encoder_config, train_stats=stats)
    # One actual optimization update before saving.
    x = torch.randn(2, 32, 3)
    optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=.001)
    pretrainer(x)["loss"].backward()
    optimizer.step()
    path = tmp_path / "encoder.pt"
    torch.save(dict(method="arctan", encoder_config=encoder_config,
                    encoder_state=pretrainer.encoder.state_dict()), path)
    loaded = load_pretrained_encoder(path).eval()
    patches, observed = loaded.patchify(x)
    h, valid, _ = loaded.encode_patches(patches, observed, torch.ones(2, loaded.feature_channels, 8))
    actual = loaded(x)
    torch.testing.assert_close(actual["features"], h.reshape(2, loaded.feature_channels, 8, 16))
    torch.testing.assert_close(actual["features"], pretrainer.encoder.eval()(x)["features"])
    assert not hasattr(loaded, "decoder")
    torch.testing.assert_close(loaded.mean, torch.tensor(stats["mean"]).float())
    road_model = PatchTSTRoadModel(loaded, dropout=0)
    output = road_model(x)
    assert output["roughness"].shape == output["disturbance_logit"].shape == (2, 8)
    (output["roughness"].sum() + output["disturbance_logit"].sum()).backward()
    assert loaded.projection[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("method", ["droppatch", "arctan"])
def test_local_corruption_rng_and_repeatable_validation(method):
    model = build_pretrainer(method, config()).eval()
    batch = dict(x=torch.randn(2, 32, 3), mask=torch.ones(2, 32, 3, dtype=torch.bool))
    state = torch.get_rng_state().clone()
    first = run_epoch(model, [batch], torch.device("cpu"), torch.Generator().manual_seed(5))
    second = run_epoch(model, [batch], torch.device("cpu"), torch.Generator().manual_seed(5))
    assert first == second
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("method", ["droppatch", "arctan"])
def test_fixed_reconstruction_problem_is_learnable(method):
    model = build_pretrainer(method, config())
    time = torch.arange(32).float() / 8
    x = torch.stack((time.sin(), time.cos(), .3 * time), dim=-1).unsqueeze(0).repeat(2, 1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0)
    losses = []
    for _ in range(45):
        optimizer.zero_grad()
        result = model(x, generator=torch.Generator().manual_seed(17))
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(result["loss"].item())
    assert losses[-1] < .55 * losses[0], losses


def test_sequence_timesteps_and_token_timesteps_have_intended_scope():
    encoder = ArcTanEncoder(**config(), channel_mode="independent")
    x = torch.randn(2, 32, 3)
    sequence = ArcTanPretrainer(encoder, timestep_mode="sequence")(x)["timesteps"]
    assert torch.equal(sequence, sequence[:, :1, :1].expand_as(sequence))
    tokens = ArcTanPretrainer(encoder, timestep_mode="token")(x)["timesteps"]
    assert tokens.unique().numel() == tokens.numel()


def test_bad_masks_and_timesteps_fail_clearly():
    x = torch.randn(2, 32, 3)
    drop = DropPatchPretrainer(PatchTST(**config()))
    with pytest.raises(ValueError, match="duplicate"):
        drop(x, kept_indices=torch.zeros(2, 3, 2, dtype=torch.long))
    diffusion = ArcTanPretrainer(ArcTanEncoder(**config()))
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        diffusion(x, timesteps=torch.full((2, 1, 8), 1.01))
