"""RoadSens labels, duplicated releases, missing modalities and source filters."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import default_collate

from road_training.dataset import CHANNELS, KAGGLE_CLASSES, RoadDataset
from road_training.patchtst import PatchTST, PatchTSTRoadModel
from road_training.tools.prepare_data import checksum, training_statistics, write_json, write_record
from road_training.tools.prepare_roadsens import INPUT_COLUMNS, construct_labels, convert, prepare
from road_training.train_multitask import joint_loss, patch_targets


def table(n=32):
    frame = pd.DataFrame({c: np.arange(n, dtype=float)/10+j for j, c in enumerate(INPUT_COLUMNS)})
    frame["seconds_elapsed"] = .09+np.arange(n)/100
    frame["annotation_text"] = ""
    frame["annotation_millisecond_press_duration"] = np.nan
    return frame


class RoadSensTests(unittest.TestCase):
    def test_closed_events_open_tail_and_blank_background(self):
        frame = table()
        frame.loc[4:11, "annotation_text"] = "Pothole"
        frame.loc[[4, 11], "annotation_millisecond_press_duration"] = [60, 80]
        frame.loc[20:, "annotation_text"] = "Bump"
        frame.loc[20, "annotation_millisecond_press_duration"] = 90
        labels, types, events = construct_labels(frame)
        np.testing.assert_array_equal(labels[4:12, :2], 1)
        self.assertTrue((labels[:4] == -100).all())
        self.assertTrue((labels[12:] == -100).all())
        self.assertTrue((labels[:, 2:] == -100).all())
        np.testing.assert_array_equal(types[4:12], 2)
        self.assertEqual([e["admitted"] for e in events], [True, False])
        self.assertEqual(events[0]["duration_seconds"], .08)  # Not 60/80 ms button duration.

    def test_unknown_types_and_multiple_button_intervals_remain_unknown(self):
        frame = table()
        frame.loc[4:15, "annotation_text"] = "Bump"
        frame.loc[[4, 8, 15], "annotation_millisecond_press_duration"] = 50
        frame.loc[20:25, "annotation_text"] = "unrecognized"
        frame.loc[[20, 25], "annotation_millisecond_press_duration"] = 50
        self.assertTrue((construct_labels(frame)[0] == -100).all())

    def test_normal_labels_do_not_invent_iri_or_kaggle_types(self):
        frame = table().drop(columns=["annotation_text", "annotation_millisecond_press_duration"])
        x, mask, time, labels, types, events = convert(frame, True)
        np.testing.assert_array_equal(labels[:, :2], 0)
        self.assertTrue((labels[:, 2:] == -100).all())
        self.assertTrue((types == 0).all())
        self.assertFalse(events)
        self.assertFalse(mask[:, 6].any())
        self.assertTrue((x[:, 6] == 0).all())
        np.testing.assert_array_equal(x[:, :6], frame[INPUT_COLUMNS].to_numpy(np.float32))
        np.testing.assert_array_equal(time, frame.seconds_elapsed.to_numpy())

    def test_missing_imu_does_not_fill_from_future(self):
        frame = table()
        frame.loc[5, "totalAcceleration_x"] = np.nan
        frame.loc[6, "gyroscope_y"] = np.inf
        x, mask, _, labels, *_ = convert(frame, True)
        self.assertEqual(x[5, 0], 0)
        self.assertFalse(mask[5, 0])
        self.assertFalse(mask[6, 4])
        self.assertTrue((labels[5:7] == -100).all())
        np.testing.assert_array_equal(x[7, :6], frame[INPUT_COLUMNS].iloc[7].to_numpy(np.float32))

    def test_gaps_reordering_and_duplicate_times_are_rejected(self):
        for index, value in [(5, .9), (5, .1), (5, np.nan)]:
            frame = table()
            frame.loc[index, "seconds_elapsed"] = value
            with self.assertRaisesRegex(ValueError, "continuous"):
                convert(frame)

    def test_import_keeps_old_splits_deduplicates_and_fits_only_train(self):
        # Temp data remains under this project, as requested for this workspace.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            base, source, out = root/"base", root/"source", root/"new"
            base.mkdir(); source.mkdir()
            records = []
            for name, dataset, source_name, split, offset in [
                ("kaggle_train", "kaggle", "real", "train", 1),
                ("lira_train", "lira_cd", "real", "train", 2),
                ("synthetic_train", "simulation", "synthetic", "train", 3),
                ("kaggle_val", "kaggle", "real", "val", 10000),
                ("lira_test", "lira_cd", "real", "test", 20000),
            ]:
                x = np.full((32, 7), offset, np.float32)
                labels = np.full((32, 5), -100, np.int16)
                records.append(write_record(base, dict(id=name, source=source_name, dataset=dataset,
                    split=split, groups=[name]), x, np.ones_like(x, bool), np.arange(32)/100,
                    labels, np.full(32, np.nan), {}))
            manifest = dict(channels=CHANNELS, sample_rate_hz=100, records=records, sources_policy="fixture",
                label_classes={"kaggle_type": KAGGLE_CLASSES},
                train_statistics={s: training_statistics(base, records, s) for s in ("real", "synthetic", "both")})
            write_json(base/"manifest.json", manifest)
            before = (base/"manifest.json").read_bytes()
            frame = table().drop(columns=["annotation_text", "annotation_millisecond_press_duration"])
            inputs = []
            for name in ("9.csv", "37.csv"):
                path = source/name
                frame.to_csv(path, index=False)
                inputs.append(dict(path=str(path), sha256=checksum(path), category="normal"))
            protocol = dict(inputs=inputs, license={"name": "CC BY 4.0"})
            write_json(source/"source_protocol.json", protocol)
            audit = prepare(base, source, out, protocol)
            self.assertEqual(audit["totals"]["recordings"], 1)
            self.assertEqual(len(audit["duplicates"]), 1)
            self.assertEqual((base/"manifest.json").read_bytes(), before)
            built = json.loads((out/"manifest.json").read_text())
            for record in records:
                current = next(r for r in built["records"] if r["id"] == record["id"])
                self.assertEqual(current, record)
                self.assertTrue((out/current["path"]).is_symlink())
            for split in ("val", "test"):
                self.assertEqual([r for r in built["records"] if r["split"] == split],
                                 [r for r in records if r["split"] == split])
            rd = RoadDataset(out, source="real", real_dataset="roadsens", window_size=32, return_labels=True)
            kd = RoadDataset(out, source="real", real_dataset="kaggle", window_size=32)
            ld = RoadDataset(out, source="real", real_dataset="lira", window_size=32)
            self.assertEqual([r["id"] for r in kd.records], ["kaggle_train"])
            self.assertEqual([r["id"] for r in ld.records], ["lira_train"])
            self.assertEqual(rd[0]["dataset"], "roadsens")
            self.assertEqual(rd.train_stats["recording_ids"], ["roadsens_009"])
            self.assertEqual(rd.train_stats["count"][6], 0)
            self.assertEqual(rd.train_stats["mean"][6], 0)
            self.assertEqual(rd.train_stats["std"][6], 1)
            heldout = RoadDataset(out, source="real", split="val", window_size=32)
            self.assertNotIn("kaggle_val", heldout.train_stats["recording_ids"])
            self.assertLess(max(heldout.train_stats["mean"]), 10)
            with self.assertRaisesRegex(ValueError, "No complete windows"):
                RoadDataset(out, source="real", real_dataset="roadsens", split="val", window_size=32)
            # Both patch heads accept the missing speed and IRI. Only detection trains.
            batch = default_collate([rd[0]])
            model = PatchTSTRoadModel(PatchTST(patch_length=4, d_model=16, n_heads=2,
                n_layers=1, ffn_dim=32, max_patches=8, train_stats=rd.train_stats))
            targets = patch_targets(batch, 4)
            self.assertFalse(targets["roughness_valid"].any())
            loss = joint_loss(model(batch["x"], batch["mask"]), targets)
            loss["total"].backward()
            self.assertTrue(torch.isfinite(loss["total"]))
            self.assertTrue(all(p.grad is None for p in model.roughness_head.parameters()))
            self.assertTrue(any(p.grad is not None for p in model.disturbance_head.parameters()))
            with self.assertRaises(FileExistsError):
                prepare(base, source, out, protocol)


if __name__ == "__main__":
    unittest.main()
