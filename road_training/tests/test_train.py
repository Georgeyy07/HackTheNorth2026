"""Supervised masking, aggregate metrics, validation isolation and updates."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from road_training.train import (CLASSES, balanced_alpha, classification_metrics, focal_loss,
                   group_patch_targets, majority_patch_targets, labeled_windows, run_epoch, parse_args)
from road_training.patchtst import PatchTST, PatchTSTPredictor


class LogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0
        self.encoder = SimpleNamespace(patch_length=1)

    def forward(self, x, observed):
        self.calls += 1
        return x.reshape(x.shape[0], -1, self.encoder.patch_length, x.shape[-1]).mean(2) * self.scale


def batch(logits, labels, observed=None):
    x = torch.tensor([logits], dtype=torch.float32)
    mask = torch.ones_like(x, dtype=torch.bool)
    if observed is not None:
        mask &= torch.tensor([observed], dtype=torch.bool).unsqueeze(-1)
    return {"x": x, "mask": mask, "labels": {"defect": torch.tensor([labels])}}


class TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_default_training_uses_cuda_bfloat16(self):
        args = parse_args([])
        self.assertEqual((args.device, args.precision, args.stride), ("cuda", "bf16", 1))
        cpu = parse_args(["--device", "cpu", "--precision", "fp32"])
        self.assertEqual(cpu.precision, "fp32")
        with self.assertRaisesRegex(ValueError, "on CUDA"):
            run_epoch(LogitModel(), [], "defect", "cpu", precision="bf16")

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 required")
    def test_cuda_bfloat16_train_and_validation_keep_fp32_weights_and_finite_loss(self):
        torch.manual_seed(13)
        model = PatchTSTPredictor(PatchTST(patch_length=4, d_model=16, n_heads=2,
                                          n_layers=1, ffn_dim=32, max_patches=4), 5).cuda()
        x = torch.randn(2, 16, 7)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[0, :, 0] = False  # Fully missing channel must also work in BF16.
        x[~mask] = float("nan")
        item = {"x": x, "mask": mask, "labels": {"kaggle_type":
                torch.tensor([[0, 1, 2, 3], [4, 0, 1, 4]]).repeat_interleave(4, dim=1)}}
        activation_dtypes = []
        hook = model.encoder.register_forward_hook(
            lambda module, inputs, output: activation_dtypes.append(output["features"].dtype))
        self.addCleanup(hook.remove)
        before = model.head[-1].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        train = run_epoch(model, [item], "kaggle_type", "cuda", optimizer, precision="bf16")
        self.assertFalse(torch.equal(before, model.head[-1].weight))
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        after = {name: value.clone() for name, value in model.state_dict().items()}
        val = run_epoch(model, [item], "kaggle_type", "cuda", precision="bf16")
        self.assertEqual(activation_dtypes, [torch.bfloat16, torch.bfloat16])
        self.assertFalse(model.training)
        for metrics in (train, val):
            self.assertEqual(metrics["labeled_patches"], 8)
            self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
            self.assertGreaterEqual(metrics["macro_f1"], 0)
            self.assertLessEqual(metrics["macro_f1"], 1)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, after[name], rtol=0, atol=0)

    def test_patch_grouping_preserves_boundaries_unknowns_and_input_mask(self):
        item = batch([[0, 0]] * 8, [0, 0, 1, 1, 1, -100, 4, 0],
                     [True, True, True, False, True, True, True, True])
        labels, keep = group_patch_targets(item, "defect", 4)
        self.assertEqual(labels.shape, (1, 2, 4))
        self.assertEqual(labels.tolist(), [[[0, 0, 1, 1], [1, -100, 4, 0]]])
        self.assertEqual(keep.tolist(), [[[True, True, True, False], [True, False, True, True]]])
        torch.testing.assert_close(labels.flatten(1), item["labels"]["defect"])
        for invalid in (0, 3, 2.5):
            with self.assertRaises(ValueError):
                group_patch_targets(item, "defect", invalid)

    def test_grouped_focal_matches_flat_loss_and_gradients(self):
        torch.manual_seed(12)
        logits = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
        labels = torch.randint(5, (2, 3, 4))
        labels[0, 1, 2] = -100
        alpha = torch.tensor([.2, 4, 2, 3, 1], dtype=torch.float64)
        grouped = focal_loss(logits, labels, alpha=alpha)
        flat_logits = logits.detach().reshape(-1, 5).clone().requires_grad_()
        flat = focal_loss(flat_logits, labels.reshape(-1), alpha=alpha)
        self.assertEqual(grouped.shape, labels.shape)
        torch.testing.assert_close(grouped.reshape(-1), flat)
        grouped.sum().backward()
        flat.sum().backward()
        torch.testing.assert_close(logits.grad.reshape(-1, 5), flat_logits.grad)
        self.assertEqual(logits.grad[0, 1, 2].abs().sum().item(), 0)

    def test_epoch_scores_each_patch_once_and_ignores_tied_votes(self):
        item = batch([[3, 0]] * 8, [0, 0, 0, 1, 0, 0, 1, 1])
        model = LogitModel()
        model.encoder.patch_length = 4
        metrics = run_epoch(model, [item], "defect", "cpu")
        self.assertEqual(metrics["labeled_patches"], 1)
        self.assertEqual(metrics["metric_unit"], "patch")
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [0, 0]])
        self.assertAlmostEqual(metrics["loss"], focal_loss(torch.tensor([[3., 0.]]), torch.tensor([0])).item())

    def test_majority_votes_include_normal_and_ignore_unknown_invalid_and_ties(self):
        # Patch 37's exact 2-normal/14-manhole vote, then a normal majority,
        # a tie, an unknown patch, and a patch with only one valid known vote.
        y = [[0, 0] + [1] * 14, [0] * 9 + [4] * 7,
             [0] * 8 + [1] * 8, [-100] * 16, [2] + [4] * 15]
        labels = torch.tensor([sum(y, [])])
        observed = torch.ones(1, 80, 7, dtype=torch.bool)
        observed[:, 65:] = False
        item = {"labels": {"kaggle_type": labels}, "mask": observed}
        majority, keep = majority_patch_targets(item, "kaggle_type", 16)
        self.assertEqual(majority.tolist(), [[1, 0, -100, -100, 2]])
        self.assertEqual(keep.tolist(), [[True, True, False, False, True]])

    def test_vote_uses_unique_most_frequent_class_and_excludes_unknown_ids(self):
        item = {"labels": {"kaggle_type": torch.tensor([[1] * 6 + [2] * 5 + [3] * 5,
                                                         [4] + [-100] * 15])},
                "mask": torch.ones(2, 16, 7, dtype=torch.bool)}
        labels, keep = majority_patch_targets(item, "kaggle_type", 16)
        self.assertEqual(labels.tolist(), [[1], [4]])
        self.assertTrue(keep.all())
        item["labels"]["kaggle_type"][0, 0] = 5
        with self.assertRaisesRegex(ValueError, "Invalid class ID"):
            majority_patch_targets(item, "kaggle_type", 16)

    def test_disturbance_votes_combine_types_before_majority(self):
        types = torch.tensor([[0]*7 + [1]*5 + [4]*4])
        item = {"labels": {"kaggle_type": types, "localized_disturbance": (types > 0).long()},
                "mask": torch.ones(1, 16, 7, dtype=torch.bool)}
        original, _ = majority_patch_targets(item, "kaggle_type", 16)
        combined, keep = majority_patch_targets(item, "localized_disturbance", 16)
        self.assertEqual(original.item(), 0)
        self.assertEqual(combined.item(), 1)
        self.assertTrue(keep.item())
        self.assertEqual(parse_args(["--target", "localized_disturbance"]).target, "localized_disturbance")

    def test_fixed_taxonomy_reports_absent_class_and_binary_positive_f1(self):
        result = classification_metrics(torch.tensor([[2, 1], [1, 0]]), ("normal", "defect"))
        self.assertAlmostEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertEqual(result["per_class"]["defect"]["f1"], 0)
        absent = classification_metrics(torch.tensor([[0, 0], [0, 4]]), ("normal", "defect"))
        self.assertEqual(absent["per_class"]["normal"]["support"], 0)
        self.assertEqual(absent["macro_f1"], .5)

    def test_validation_masks_unknown_and_missing_inputs_and_aggregates_samples(self):
        first = batch([[3, 0], [0, 3], [3, 0], [100, 0]], [0, 1, -100, 1],
                      [True, True, True, False])
        second = batch([[0, 3]], [0])
        model = LogitModel().train()
        before = model.scale.detach().clone()
        alpha = torch.tensor([.5, 3.])
        result = run_epoch(model, [first, second], "defect", "cpu", alpha=alpha)
        ce = F.cross_entropy(torch.tensor([[3., 0.], [0., 3.], [0., 3.]]),
                             torch.tensor([0, 1, 0]), reduction="none")
        expected_loss = (alpha[torch.tensor([0, 1, 0])] * (1 - ce.exp().reciprocal()).square() * ce).mean().item()
        self.assertAlmostEqual(result["loss"], expected_loss, places=6)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 1]])
        self.assertEqual(result["labeled_patches"], 3)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        torch.testing.assert_close(model.scale, before)
        self.assertIsNone(model.scale.grad)
        self.assertFalse(model.training)

    def test_focal_gamma_zero_recovers_cross_entropy_and_ignores_unknown(self):
        logits = torch.tensor([[2., -1.], [-1., 1.], [10., -10.]], requires_grad=True)
        labels = torch.tensor([0, 1, -100])
        actual = focal_loss(logits, labels, gamma=0)
        expected = F.cross_entropy(logits, labels, reduction="none")
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        torch.testing.assert_close(logits.grad[-1], torch.zeros(2))

    def test_focal_weights_apply_after_probability_and_focus_on_hard_samples(self):
        # Known class probabilities avoid computing the expected answer by
        # repeating the implementation's cross-entropy calculation.
        probabilities = torch.tensor([[.9, .1], [.9, .1]], dtype=torch.float64)
        labels = torch.tensor([0, 1])
        alpha = torch.tensor([.5, 3.], dtype=torch.float64)
        pt = torch.tensor([.9, .1], dtype=torch.float64)
        actual = focal_loss(probabilities.log(), labels, gamma=2, alpha=alpha)
        expected = alpha * (1 - pt).square() * -pt.log()
        torch.testing.assert_close(actual, expected)
        self.assertLess((actual / (alpha * -pt.log()))[0].item(), .011)
        self.assertGreater((actual / (alpha * -pt.log()))[1].item(), .8)

    def test_balanced_alpha_uses_counts_and_keeps_unseen_classes_neutral(self):
        counts = torch.tensor([900., 100., 0.])
        alpha = balanced_alpha(counts)
        torch.testing.assert_close(alpha, torch.tensor([1000/1800, 5., 1.]))
        self.assertAlmostEqual((alpha * counts).sum().item() / counts.sum().item(), 1)
        for invalid in ([0, 0], [-1, 3], [float("nan"), 1]):
            with self.assertRaises(ValueError):
                balanced_alpha(invalid)

    def test_focal_extreme_logits_have_finite_loss_and_gradients(self):
        logits = torch.tensor([[1000., -1000.], [-1000., 1000.]], requires_grad=True)
        loss = focal_loss(logits, torch.tensor([1, 1]), alpha=torch.tensor([.5, 3.])).sum()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_five_class_metrics_separate_detection_from_type_errors(self):
        # Correct defect detection with every defect assigned the wrong type.
        confusion = torch.tensor([[3, 0, 0, 0, 0], [0, 0, 2, 0, 0],
                                  [0, 0, 0, 2, 0], [0, 0, 0, 0, 2], [0, 2, 0, 0, 0]])
        result = classification_metrics(confusion, CLASSES["kaggle_type"])
        self.assertAlmostEqual(result["macro_f1"], .2)
        self.assertEqual(result["binary_defect"]["confusion_matrix"], [[3, 0], [0, 8]])
        self.assertEqual(result["binary_defect"]["per_class"]["defect"]["f1"], 1)

    def test_normal_and_all_four_types_contribute_to_five_class_loss(self):
        logits = torch.eye(5).unsqueeze(0) * 3
        item = {"x": logits, "mask": torch.ones_like(logits, dtype=torch.bool),
                "labels": {"kaggle_type": torch.arange(5).unsqueeze(0)}}
        model = LogitModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        metrics = run_epoch(model, [item], "kaggle_type", "cpu", optimizer)
        self.assertEqual(metrics["labeled_patches"], 5)
        self.assertEqual(metrics["macro_f1"], 1)
        self.assertEqual(metrics["per_class"]["normal"]["support"], 1)
        self.assertGreater(model.scale.item(), 1)

    def test_training_updates_parameters_and_skips_unlabeled_batch(self):
        model = LogitModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        labeled = batch([[3, 0], [0, 3]], [0, 1])
        unknown = batch([[0, 3]], [-100])
        before = run_epoch(model, [labeled], "defect", "cpu")["loss"]
        model.calls = 0
        trained = run_epoch(model, [unknown, labeled], "defect", "cpu", optimizer)
        self.assertEqual(model.calls, 1)
        self.assertTrue(model.training)
        self.assertEqual(trained["labeled_patches"], 2)
        self.assertGreater(model.scale.item(), 1)
        after = run_epoch(model, [labeled], "defect", "cpu")["loss"]
        self.assertLess(after, before)
        with self.assertRaisesRegex(ValueError, "no usable targets"):
            run_epoch(model, [unknown], "defect", "cpu")

    def test_subset_keeps_labeled_windows_without_relabeling_unknown_samples(self):
        class Samples(list):
            source, split = "real", "train"

        examples = [batch([[1, 0], [0, 1]], [0, -100]),
                    batch([[1, 0]], [1], [False]), batch([[0, 1]], [1])]
        # Dataset items have no batch dimension.
        data = Samples({"x": b["x"][0], "mask": b["mask"][0],
                        "labels": {"defect": b["labels"]["defect"][0]}} for b in examples)
        subset, counts = labeled_windows(data, "defect", 1)
        self.assertEqual(subset.indices, [0, 2])
        self.assertEqual(counts, [1, 1])
        self.assertEqual(subset[0]["labels"]["defect"].tolist(), [0, -100])
        with self.assertRaisesRegex(ValueError, "No usable defect labels"):
            labeled_windows(Samples([data[1]]), "defect", 1)

    def test_focal_class_counts_are_majority_patches_not_source_samples(self):
        class Samples(list):
            source, split = "real", "train"

        data = Samples({"mask": torch.ones(4, 7, dtype=torch.bool),
                        "labels": {"defect": torch.tensor(y)}}
                       for y in ([0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [-100] * 4))
        subset, counts = labeled_windows(data, "defect", 4)
        self.assertEqual(subset.indices, [0, 2])
        self.assertEqual(counts, [1, 1])


if __name__ == "__main__":
    unittest.main()
