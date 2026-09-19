"""Read live TensorBoard events before writer close; compare to epoch metrics."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from road_training.train import CLASSES, classification_metrics, write_epoch_metrics


@unittest.skipUnless(importlib.util.find_spec("tensorboard"), "Optional tensorboard package required")
class MonitoringTests(unittest.TestCase):
    def test_train_val_curves_are_flushed_at_epoch_steps_and_separate_binary_f1(self):
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        with tempfile.TemporaryDirectory() as folder:
            for split in ("train", "val"):
                directory = Path(folder) / split
                with SummaryWriter(str(directory)) as writer:
                    expected = []
                    for epoch in (1, 2):
                        # Defects detected but misclassified: binary F1 must differ from type macro F1.
                        counts = torch.tensor([[3, 0, 0, 0, 0], [0, 0, 2, 0, 0],
                                               [0, 0, 0, 2, 0], [0, 0, 0, 0, 2], [0, 2, 0, 0, 0]])
                        counts += torch.eye(5, dtype=torch.long) * (epoch + (split == "train"))
                        metrics = classification_metrics(counts, CLASSES["kaggle_type"])
                        metrics["loss"] = 1 / epoch + (split == "val")
                        expected.append(metrics)
                        write_epoch_metrics(writer, epoch, metrics)
                        events = EventAccumulator(str(directory)).Reload()  # Writer still open.
                        for tag, values in {
                            "loss/focal": [m["loss"] for m in expected],
                            "f1/macro": [m["macro_f1"] for m in expected],
                            "f1/binary_defect": [1.] * epoch,
                            "f1_per_class/crack": [m["per_class"]["crack"]["f1"] for m in expected],
                        }.items():
                            points = events.Scalars(tag)
                            self.assertEqual([p.step for p in points], list(range(1, epoch + 1)))
                            for point, value in zip(points, values):
                                self.assertAlmostEqual(point.value, value, places=6)

    def test_binary_target_logs_positive_class_f1(self):
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        metrics = classification_metrics(torch.tensor([[8, 1], [1, 0]]), CLASSES["defect"])
        metrics["loss"] = .5
        with tempfile.TemporaryDirectory() as folder:
            with SummaryWriter(folder) as writer:
                write_epoch_metrics(writer, 1, metrics)
                events = EventAccumulator(folder).Reload()
                self.assertEqual(events.Scalars("f1/binary_defect")[0].value, 0)
                self.assertGreater(events.Scalars("f1/macro")[0].value, 0)

    def test_disturbance_logs_positive_f1_separately_from_macro(self):
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        metrics = classification_metrics(torch.tensor([[8, 1], [1, 0]]), CLASSES["localized_disturbance"])
        metrics["loss"] = .5
        with tempfile.TemporaryDirectory() as folder:
            with SummaryWriter(folder) as writer:
                write_epoch_metrics(writer, 1, metrics)
                events = EventAccumulator(folder).Reload()
                self.assertEqual(events.Scalars("f1/localized_disturbance")[0].value, 0)
                self.assertGreater(events.Scalars("f1/macro")[0].value, 0)


if __name__ == "__main__":
    unittest.main()
