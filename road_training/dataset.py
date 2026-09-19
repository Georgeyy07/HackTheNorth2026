"""Fixed-length sensor windows. Only NumPy and PyTorch are required."""
from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CHANNELS = ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "speed")
LABELS = ("defect", "defect_assumed_normal", "kaggle_type", "synthetic_type", "quality_grade")
KAGGLE_CLASSES = ("normal", "manhole", "depression", "bump", "crack")
REAL_DATASETS = ("all", "kaggle", "lira", "roadsens")


def collection_name(record):
    """Keep dataset identity explicit when combining several real sources."""
    if record["source"] != "real":
        return "synthetic"
    name = record.get("dataset", "kaggle")
    return {"lira_cd": "lira", "roadsens4m": "roadsens"}.get(name, name)


class RoadDataset(Dataset):
    """Read 100 Hz windows without crossing recording or split boundaries.

    x and mask have shape [window_size, 7]. Values are raw SI units; missing
    inputs are zero with mask=False. No normalization, patching or augmentation
    happens here. Optional classification labels use -100 for unknown values.
    real_dataset selects all real data, Kaggle, LiRA, or RoadSens; synthetic
    selection is controlled independently by source. train_stats matches both
    filters and always comes from TRAIN, even when reading held-out windows.
    """

    def __init__(self, root=None, *, split="train", source="both",
                 window_size=1024, stride=None, return_labels=False, real_dataset="all",
                 max_cached_recordings=None):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'")
        if source not in ("real", "synthetic", "both"):
            raise ValueError("source must be 'real', 'synthetic', or 'both'")
        if real_dataset not in REAL_DATASETS:
            raise ValueError(f"real_dataset must be one of {REAL_DATASETS}")
        stride = window_size if stride is None else stride
        if type(window_size) is not int or type(stride) is not int or min(window_size, stride) < 1:
            raise ValueError("window_size and stride must be positive integers")
        if max_cached_recordings is None:
            max_cached_recordings = 128 if source == "synthetic" else 8
        if type(max_cached_recordings) is not int or max_cached_recordings < 1:
            raise ValueError("max_cached_recordings must be a positive integer")
        self.max_cached_recordings = max_cached_recordings

        self.root = Path(root) if root is not None else Path(__file__).parent / "data"
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest["sample_rate_hz"] != 100 or tuple(manifest["channels"]) != CHANNELS:
            raise ValueError("Expected the prepared 100 Hz, seven-channel dataset")
        if return_labels and tuple(manifest.get("label_classes", {}).get("kaggle_type", ())) != KAGGLE_CLASSES:
            raise ValueError("Expected five-class kaggle_type labels (normal + four defects). "
                             "Rebuild older four-class data with tools/prepare_data.py.")
        self.window_size, self.stride = window_size, stride
        self.return_labels = return_labels
        self.source, self.split = source, split
        self.real_dataset = real_dataset
        self.sample_rate_hz = manifest["sample_rate_hz"]
        # Always TRAIN statistics, even when this instance loads val or test.
        self.train_stats = manifest["train_statistics"][source]
        if real_dataset != "all" and source != "synthetic":
            options = manifest.get("train_statistics_by_real_dataset", {})
            if real_dataset in options:
                self.train_stats = options[real_dataset][source]
            elif real_dataset != "kaggle" or any(
                    collection_name(r) not in ("kaggle", "synthetic") for r in manifest["records"]):
                raise ValueError("Missing per-dataset TRAIN statistics; rebuild the dataset import")
        self.records = []
        self._ends = []
        self._cache = OrderedDict()
        total = 0
        for record in manifest["records"]:
            if record["split"] != split or (source != "both" and record["source"] != source):
                continue
            if record["source"] == "real" and real_dataset != "all":
                if collection_name(record) != real_dataset:
                    continue
            count = max(0, (record["samples"] - window_size) // stride + 1)
            if count:
                self.records.append(record)
                total += count
                self._ends.append(total)
        if not total:
            raise ValueError(f"No complete windows for split={split!r}, source={source!r}, "
                             f"window_size={window_size}. Check the manifest or use a shorter window.")

    def __len__(self):
        return self._ends[-1]

    def _open(self, record_index):
        """Bound open memory maps; larger synthetic caches avoid tiny-file churn."""
        if record_index not in self._cache:
            folder = self.root / self.records[record_index]["path"]
            names = ["x", "mask", "time"]
            if self.return_labels:
                names += ["labels", "iri"]
                roughness_files = [(folder / f"{name}.npy").exists()
                                   for name in ("overall_iri", "roughness_section")]
                if any(roughness_files) and not all(roughness_files):
                    raise ValueError(f"Incomplete overall roughness targets in {folder}")
                if all(roughness_files):
                    names += ["overall_iri", "roughness_section"]
            self._cache[record_index] = {
                name: np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                for name in names
            }
            if len(self._cache) > self.max_cached_recordings:
                self._cache.popitem(last=False)
        self._cache.move_to_end(record_index)
        return self._cache[record_index]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        record_index = bisect_right(self._ends, index)
        previous_end = self._ends[record_index - 1] if record_index else 0
        start = (index - previous_end) * self.stride
        stop = start + self.window_size
        record = self.records[record_index]
        arrays = self._open(record_index)
        # Copy this window, not the whole recording. The returned tensors are
        # writable, so an in-place augmentation cannot modify the stored data.
        result = {
            name: torch.from_numpy(np.array(arrays[name][start:stop], copy=True))
            for name in ("x", "mask", "time")
        }
        collection = collection_name(record)
        result.update(recording_id=record["id"], source=record["source"], dataset=collection, start=start)
        if self.return_labels:
            labels = np.array(arrays["labels"][start:stop], dtype=np.int64, copy=True)
            iri = np.array(arrays["iri"][start:stop], copy=True)
            result["labels"] = {name: torch.from_numpy(labels[:, j]) for j, name in enumerate(LABELS)}
            # All real defect types / all synthetic injected kinds are positive,
            # including overlapping kinds. Keep the source type labels intact.
            # Real negatives outside annotations are assumed, not verified;
            # wholly unlabeled drives and invalid intervals remain unknown.
            result["labels"]["localized_disturbance"] = result["labels"]["defect_assumed_normal"].clone()
            result["labels"]["iri_valid"] = torch.from_numpy(np.isfinite(iri))
            result["labels"]["iri"] = torch.from_numpy(np.nan_to_num(iri, nan=0.0))
            # Optional full-profile IRI targets. Never substitute background IRI
            # or invent roughness labels for unlabeled real recordings.
            overall = (np.array(arrays["overall_iri"][start:stop], copy=True)
                       if "overall_iri" in arrays else np.full(self.window_size, np.nan, np.float32))
            section = (np.array(arrays["roughness_section"][start:stop], dtype=np.int64, copy=True)
                       if "roughness_section" in arrays else np.full(self.window_size, -1, np.int64))
            result["labels"]["overall_iri_valid"] = torch.from_numpy(np.isfinite(overall) & (section >= 0))
            result["labels"]["overall_iri"] = torch.from_numpy(np.nan_to_num(overall, nan=0.0))
            result["labels"]["roughness_section"] = torch.from_numpy(section)
        return result

    def __getstate__(self):
        # Spawned DataLoader workers open their own maps rather than pickling
        # mapped arrays into each worker. This also works with num_workers=0.
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state
