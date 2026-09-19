"""Focused tests for boundaries, masks, train-only statistics and DataLoader use."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from road_training.dataset import CHANNELS, KAGGLE_CLASSES, LABELS, RoadDataset
from road_training.tools.prepare_data import kaggle_labels, training_statistics


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = []
        for name, source, split, count, offset in [
            ("real_train", "real", "train", 18, 0),
            ("synthetic_train", "synthetic", "train", 12, 50),
            ("heldout", "real", "val", 18, 10000),
        ]:
            folder = self.root / name
            folder.mkdir()
            x = np.repeat((np.arange(count) + offset)[:, None], 7, axis=1).astype(np.float32)
            mask = np.ones_like(x, bool)
            mask[1, 0] = False
            x[1, 0] = 0
            np.save(folder / "x.npy", x)
            np.save(folder / "mask.npy", mask)
            np.save(folder / "time.npy", np.arange(count) / 100)
            np.save(folder / "labels.npy", np.full((count, len(LABELS)), -100, np.int16))
            np.save(folder / "iri.npy", np.full(count, np.nan, np.float32))
            self.records.append(dict(id=name, path=name, source=source, split=split, samples=count))
        manifest = dict(sample_rate_hz=100, channels=CHANNELS, records=self.records,
                        label_classes={"kaggle_type": KAGGLE_CLASSES},
                        train_statistics={s: training_statistics(self.root, self.records, s)
                                          for s in ["real", "synthetic", "both"]})
        (self.root / "manifest.json").write_text(json.dumps(manifest))

    def test_source_and_split_filters(self):
        real = RoadDataset(self.root, source="real", window_size=6)
        synthetic = RoadDataset(self.root, source="synthetic", window_size=6)
        both = RoadDataset(self.root, window_size=6)
        self.assertEqual((len(real), len(synthetic), len(both)), (3, 2, 5))
        self.assertTrue(all(r["split"] == "train" for r in both.records))
        val = RoadDataset(self.root, split="val", window_size=6)
        self.assertEqual(val[0]["recording_id"], "heldout")
        with self.assertRaisesRegex(ValueError, "No complete windows"):
            RoadDataset(self.root, source="synthetic", split="test", window_size=6)

    def test_real_dataset_filters_keep_matching_train_statistics_and_missing_gyros(self):
        shutil.copytree(self.root / "real_train", self.root / "lira_train")
        folder=self.root / "lira_train"
        x=np.load(folder / "x.npy");mask=np.load(folder / "mask.npy")
        x[:,:3]+=200;mask[:,3:6]=False;x[~mask]=0
        np.save(folder / "x.npy",x);np.save(folder / "mask.npy",mask)
        manifest=json.loads((self.root / "manifest.json").read_text())
        manifest["records"].append(dict(self.records[0],id="lira_train",path="lira_train",dataset="lira_cd"))
        manifest["records"][2]["dataset"]="lira_cd"  # Large held-out values must never fit normalization.
        manifest["train_statistics_by_real_dataset"]={}
        for name in ("kaggle","lira"):
            chosen=[r for r in manifest["records"] if r["source"]!="real" or (r.get("dataset")=="lira_cd")== (name=="lira")]
            manifest["train_statistics_by_real_dataset"][name]={s:training_statistics(self.root,chosen,s,allow_missing_channels=True)
                                                               for s in ("real","synthetic","both")}
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        kaggle=RoadDataset(self.root,source="real",real_dataset="kaggle",window_size=6)
        lira=RoadDataset(self.root,source="real",real_dataset="lira",window_size=6)
        both=RoadDataset(self.root,source="both",real_dataset="lira",window_size=6)
        val=RoadDataset(self.root,source="real",real_dataset="lira",split="val",window_size=6)
        self.assertEqual((len(kaggle),len(lira),len(both)),(3,3,5))
        self.assertEqual(kaggle[0]["dataset"],"kaggle");self.assertEqual(lira[0]["dataset"],"lira")
        self.assertEqual(kaggle.train_stats,manifest["train_statistics"]["real"])
        self.assertEqual(lira.train_stats,val.train_stats)
        self.assertEqual(lira.train_stats["recording_ids"],["lira_train"])
        self.assertEqual(lira.train_stats["count"][3:6],[0,0,0])
        self.assertEqual(lira.train_stats["mean"][3:6],[0.,0.,0.])
        self.assertEqual(lira.train_stats["std"][3:6],[1.,1.,1.])
        self.assertFalse(lira[0]["mask"][:,3:6].any())

    def test_no_window_crosses_recording_or_tail(self):
        ds = RoadDataset(self.root, window_size=8, stride=5)
        self.assertEqual(len(ds), 4)  # real starts 0/5/10; synthetic starts 0
        np.testing.assert_array_equal(ds[2]["x"][:, 1], np.arange(10, 18))
        np.testing.assert_array_equal(ds[3]["x"][:, 1], np.arange(50, 58))
        self.assertEqual(ds[-1]["recording_id"], "synthetic_train")
        with self.assertRaises(IndexError):
            ds[4]

    def test_returned_tensors_cannot_mutate_the_recording(self):
        ds = RoadDataset(self.root, window_size=8)
        ds[0]["x"].fill_(-999)
        self.assertEqual(ds[0]["x"][2, 1].item(), 2)
        self.assertFalse(ds[0]["mask"][1, 0].item())

    def test_unknown_labels_and_regression_mask(self):
        item = RoadDataset(self.root, window_size=8, return_labels=True)[0]
        self.assertEqual(item["labels"]["defect"].dtype, torch.int64)
        self.assertTrue((item["labels"]["defect"] == -100).all())
        self.assertFalse(item["labels"]["iri_valid"].any())
        self.assertTrue(torch.isfinite(item["labels"]["iri"]).all())
        self.assertFalse(item["labels"]["overall_iri_valid"].any())
        self.assertTrue((item["labels"]["roughness_section"] == -1).all())

    def test_overall_roughness_is_separate_from_background_iri(self):
        folder = self.root / "synthetic_train"
        np.save(folder / "overall_iri.npy", np.r_[np.nan, np.full(11, 3.)].astype(np.float32))
        np.save(folder / "roughness_section.npy", np.r_[-1, np.zeros(11)].astype(np.int32))
        labels = RoadDataset(self.root, source="synthetic", window_size=8, return_labels=True)[0]["labels"]
        self.assertEqual(labels["overall_iri"].tolist(), [0.] + [3.]*7)
        self.assertEqual(labels["overall_iri_valid"].tolist(), [False] + [True]*7)
        self.assertFalse(labels["iri_valid"].any())  # Background file is still unknown.
        (folder / "roughness_section.npy").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete overall"):
            RoadDataset(self.root, source="synthetic", window_size=8, return_labels=True)[0]

    def test_statistics_use_train_and_observed_values_only(self):
        ds = RoadDataset(self.root, source="real", split="val", window_size=8)
        values = np.delete(np.arange(18, dtype=float), 1)
        self.assertAlmostEqual(ds.train_stats["mean"][0], values.mean())
        self.assertAlmostEqual(ds.train_stats["std"][0], values.std())
        self.assertEqual(ds.train_stats["count"][0], 17)
        self.assertEqual(ds.train_stats["recording_ids"], ["real_train"])

    def test_localized_disturbance_includes_all_types_and_preserves_unknowns(self):
        time = np.arange(18, dtype=float) / 100
        annotations = [dict(rel_t_start=(i+2)/100, rel_t_end=(i+3)/100, anomaly=kind)
                       for i, kind in enumerate(KAGGLE_CLASSES[1:])]
        annotations += [dict(rel_t_start=.05, rel_t_end=.06, anomaly="manhole"),
                        dict(rel_t_start=.07, rel_t_end=.08, anomaly="unrecognized")]
        valid = np.load(self.root / "real_train/mask.npy").all(1)
        np.save(self.root / "real_train/labels.npy", kaggle_labels(time, annotations, valid))
        item = RoadDataset(self.root, source="real", window_size=12, return_labels=True)[0]
        target = item["labels"]["localized_disturbance"]
        self.assertEqual(target.tolist(), [0, -100, 1, 1, 1, 1, 0, -100, 0, 0, 0, 0])
        self.assertEqual(item["labels"]["kaggle_type"][5].item(), -100)  # Two types, one disturbance.
        target.fill_(0)
        self.assertEqual(item["labels"]["defect_assumed_normal"][5].item(), 1)

        # Synthetic type conflicts must also remain positive for presence.
        labels = np.full((12, len(LABELS)), -100, np.int16)
        labels[:5, 1] = [0, 1, 1, 1, 1]
        labels[:5, 3] = [-100, 0, 1, 2, -100]
        np.save(self.root / "synthetic_train/labels.npy", labels)
        item = RoadDataset(self.root, source="synthetic", window_size=12, return_labels=True)[0]
        self.assertEqual(item["labels"]["localized_disturbance"][:6].tolist(), [0, 1, 1, 1, 1, -100])
        unknown = RoadDataset(self.root, split="val", window_size=12, return_labels=True)[0]
        self.assertTrue((unknown["labels"]["localized_disturbance"] == -100).all())

    def test_labels_do_not_invent_normal_for_unlabeled_drives(self):
        t = np.arange(6, dtype=float)
        self.assertTrue((kaggle_labels(t, [], np.ones(6, bool)) == -100).all())
        annotations = [dict(rel_t_start=1, rel_t_end=4, anomaly="bump"),
                       dict(rel_t_start=2, rel_t_end=3, anomaly="crack")]
        labels = kaggle_labels(t, annotations, np.ones(6, bool))
        self.assertEqual(labels[0, 0], -100)
        self.assertEqual(labels[0, 1], 0)  # Explicitly named weak-label target.
        self.assertEqual(labels[2, 0], 1)  # Types disagree, binary labels agree.
        self.assertEqual(labels[2, 2], -100)
        self.assertEqual(labels[4, 0], -100)  # End boundary is exclusive.
        np.testing.assert_array_equal(labels[:, 2], [0, 3, -100, 3, 0, 0])

    def test_five_classes_include_normal_but_preserve_unknown_and_invalid(self):
        annotations = [dict(rel_t_start=i+1, rel_t_end=i+2, anomaly=kind)
                       for i, kind in enumerate(KAGGLE_CLASSES[1:])]
        annotations += [dict(rel_t_start=5, rel_t_end=6, anomaly="unrecognized"),
                        dict(rel_t_start=6, rel_t_end=7, anomaly="no_anomaly")]
        valid = np.ones(9, bool)
        valid[-1] = False
        labels = kaggle_labels(np.arange(9, dtype=float), annotations, valid)
        np.testing.assert_array_equal(labels[:, 2], [0, 1, 2, 3, 4, -100, 0, 0, -100])

    def test_old_four_class_manifest_cannot_silently_change_label_meanings(self):
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["label_classes"]["kaggle_type"] = list(KAGGLE_CLASSES[1:])
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "five-class"):
            RoadDataset(self.root, window_size=8, return_labels=True)

    def test_spawn_workers_after_parent_has_opened_maps(self):
        ds = RoadDataset(self.root, window_size=8, return_labels=True)
        ds[0]
        self.assertEqual(ds.__getstate__()["_cache"], {})
        loader = DataLoader(ds, batch_size=2, num_workers=2, multiprocessing_context="spawn")
        batches = list(loader)
        self.assertEqual(batches[0]["x"].shape, (2, 8, 7))
        np.testing.assert_array_equal(batches[0]["x"][0, :, 1], np.arange(8))

    def test_cache_capacity_changes_only_open_maps(self):
        small = RoadDataset(self.root, window_size=6, return_labels=True, max_cached_recordings=1)
        large = RoadDataset(self.root, window_size=6, return_labels=True, max_cached_recordings=8)
        for index in [0, 3, 1, 4, 0]:
            a, b = small[index], large[index]
            self.assertEqual(a["recording_id"], b["recording_id"])
            for key in ("x", "mask", "time"):
                self.assertTrue(torch.equal(a[key], b[key]))
            for key in a["labels"]:
                self.assertTrue(torch.equal(a["labels"][key], b["labels"][key]))
        self.assertEqual(len(small._cache), 1)
        self.assertEqual(len(large._cache), 2)
        self.assertEqual(RoadDataset(self.root, source="synthetic", window_size=6).max_cached_recordings, 128)

    def test_invalid_arguments_fail_clearly(self):
        for kwargs in [dict(source="typo"), dict(split="pretrain"), dict(stride=0), dict(window_size=8.5),
                       dict(max_cached_recordings=0), dict(max_cached_recordings=True)]:
            with self.assertRaises(ValueError):
                RoadDataset(self.root, **kwargs)


if __name__ == "__main__":
    unittest.main()
