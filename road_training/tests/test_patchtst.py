"""Check patch ordering, hidden-target isolation, missing data and learning."""
import io
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

from road_training.patchtst import PatchTST, PatchTSTPretrainer, PatchTSTPredictor


class PatchTSTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.config = dict(channels=3, patch_length=4, d_model=16, n_heads=2,
                           n_layers=2, ffn_dim=32, dropout=0, max_patches=16)
        self.x = torch.randn(2, 32, 3)
        self.valid = torch.ones_like(self.x, dtype=torch.bool)

    def encoder(self, **kwargs):
        return PatchTST(**self.config, **kwargs)

    def test_patch_order_and_channelwise_z_scores(self):
        encoder = self.encoder(train_stats={"mean": [0, 10, 20], "std": [1, 2, 4]})
        x = torch.arange(32).view(1, 32, 1).float() + torch.tensor([0, 10, 20])
        patches, mask = encoder.patchify(x)
        self.assertEqual(patches.shape, (1, 3, 8, 4))
        torch.testing.assert_close(patches[0, 0, 1], torch.arange(4, 8).float())
        torch.testing.assert_close(patches[0, 1, 1], torch.arange(4, 8).float() / 2)
        torch.testing.assert_close(patches[0, 2, 1], torch.arange(4, 8).float() / 4)
        self.assertTrue(mask.all())

    def test_encoder_channels_are_independent(self):
        encoder = self.encoder().eval()
        changed = self.x.clone()
        changed[:, :, 1] += 5
        with torch.no_grad():
            a = encoder(self.x)["features"]
            b = encoder(changed)["features"]
        torch.testing.assert_close(a[:, [0, 2]], b[:, [0, 2]])
        self.assertFalse(torch.allclose(a[:, 1], b[:, 1]))

    def test_hidden_values_cannot_change_predictions_or_input_gradients(self):
        model = PatchTSTPretrainer(self.encoder()).eval()
        hidden = torch.zeros(2, 3, 8, dtype=torch.bool)
        hidden[:, :, 2:4] = True
        hidden_samples = hidden.unsqueeze(-1).expand(-1, -1, -1, 4).reshape(2, 3, 32).transpose(1, 2)
        altered = self.x.masked_fill(hidden_samples, 999)
        before = model(self.x, self.valid, patch_mask=hidden)
        after = model(altered, self.valid, patch_mask=hidden)
        torch.testing.assert_close(before["prediction"], after["prediction"], rtol=0, atol=0)
        self.assertFalse(torch.equal(before["target"], after["target"]))
        x = self.x.clone().requires_grad_()
        model(x, self.valid, patch_mask=hidden)["loss"].backward()
        self.assertTrue((x.grad[hidden_samples] == 0).all())

    def test_missing_payloads_are_ignored_and_loss_is_only_on_masked_observations(self):
        model = PatchTSTPretrainer(self.encoder()).eval()
        valid = self.valid.clone()
        valid[:, 8:11, 0] = False
        valid[:, :, 2] = False
        x = self.x.masked_fill(~valid, float("nan"))
        hidden = torch.zeros(2, 3, 8, dtype=torch.bool)
        hidden[:, :, 2] = True
        output = model(x, valid, patch_mask=hidden)
        baseline = model(self.x.masked_fill(~valid, 0), valid, patch_mask=hidden)
        torch.testing.assert_close(output["prediction"], baseline["prediction"])
        expected_mask = valid.transpose(1, 2).unfold(-1, 4, 4) & hidden.unsqueeze(-1)
        self.assertTrue(torch.equal(output["loss_mask"], expected_mask))
        expected = (output["prediction"][expected_mask] - output["target"][expected_mask]).abs().mean()
        torch.testing.assert_close(output["loss"], expected)
        self.assertTrue(torch.isfinite(output["prediction"]).all())

    def test_fully_missing_batch_is_finite_with_zero_loss(self):
        model = PatchTSTPretrainer(self.encoder())
        output = model(self.x, torch.zeros_like(self.valid))
        self.assertTrue(torch.isfinite(output["prediction"]).all())
        self.assertEqual(output["loss"].item(), 0)
        self.assertFalse(output["loss_mask"].any())
        output["loss"].backward()

    def test_random_masks_keep_context_and_skip_unavailable_patches(self):
        valid = torch.tensor([[[True, True, True, False], [True, False, False, False],
                               [False, False, False, False]]])
        hidden = PatchTSTPretrainer.random_mask(valid, .99)
        self.assertEqual(hidden.sum().item(), 2)
        self.assertFalse((hidden & ~valid).any())
        self.assertEqual((valid & ~hidden)[0, 0].sum().item(), 1)

    def test_patch_sample_and_window_heads_and_supervised_backward(self):
        for level, shape in [("patch", (2, 8, 4)), ("sample", (2, 32, 4)), ("window", (2, 4))]:
            model = PatchTSTPredictor(self.encoder(), 4, output_level=level, dropout=0)
            logits = model(self.x, self.valid)
            self.assertEqual(logits.shape, shape)
            target = torch.zeros(shape[:-1], dtype=torch.long)
            F.cross_entropy(logits.reshape(-1, 4), target.reshape(-1)).backward()
            self.assertGreater(model.encoder.value_embedding.weight.grad.abs().sum().item(), 0)

    def test_default_patch_head_checkpoint_preserves_one_prediction_per_patch(self):
        model = PatchTSTPredictor(self.encoder(train_stats={"mean": [1, 2, 3], "std": [2, 3, 4]}), 5).eval()
        self.assertEqual(model.output_level, "patch")
        self.assertEqual(model.head[-1].out_features, 5)
        buffer = io.BytesIO()
        torch.save({"encoder_config": model.encoder.config, "output_level": model.output_level,
                    "state": model.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = PatchTSTPredictor(PatchTST(**saved["encoder_config"]), 5,
                                     output_level=saved["output_level"]).eval()
        restored.load_state_dict(saved["state"])
        with torch.no_grad():
            expected, actual = model(self.x, self.valid), restored(self.x, self.valid)
        self.assertEqual(actual.shape, (2, 8, 5))
        torch.testing.assert_close(actual, expected)

    def test_checkpoint_restores_normalization_and_outputs(self):
        stats = {"mean": [1, 2, 3], "std": [4, 5, 6]}
        original = self.encoder(train_stats=stats).eval()
        buffer = io.BytesIO()
        torch.save({"config": original.config, "state": original.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = PatchTST(**saved["config"]).eval()
        restored.load_state_dict(saved["state"])
        torch.testing.assert_close(original(self.x)["features"], restored(self.x)["features"])
        torch.testing.assert_close(restored.std, torch.tensor([4., 5., 6.]))

    def test_fixed_mask_reconstruction_can_be_optimized(self):
        model = PatchTSTPretrainer(self.encoder(), loss="mse")
        # A learnable repeating waveform, not independent noise whose hidden
        # values cannot be inferred from context.
        wave = torch.sin(torch.arange(32).float() * torch.pi / 2)
        x = wave.view(1, 32, 1).repeat(2, 1, 3) * torch.tensor([1., 2., 3.])
        hidden = torch.zeros(2, 3, 8, dtype=torch.bool)
        hidden[:, :, 2:4] = True
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        initial = model(x, self.valid, patch_mask=hidden)["loss"].item()
        for _ in range(25):
            optimizer.zero_grad(set_to_none=True)
            loss = model(x, self.valid, patch_mask=hidden)["loss"]
            loss.backward()
            optimizer.step()
        final = model(x, self.valid, patch_mask=hidden)["loss"].item()
        self.assertLess(final, initial * .6)

    def test_bad_shapes_fail_without_silent_padding_or_truncation(self):
        encoder = self.encoder()
        for x in [self.x[:, :31], self.x[:, :, :2], torch.zeros(2, 68, 3)]:
            with self.assertRaises(ValueError):
                encoder(x)
        with self.assertRaises(ValueError):
            encoder(self.x, self.valid.float())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_autocast_backward(self):
        model = PatchTSTPretrainer(self.encoder()).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(self.x.cuda(), self.valid.cuda())
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        self.assertTrue(torch.isfinite(model.encoder.value_embedding.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
