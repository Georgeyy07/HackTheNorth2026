"""One-time conversion of pinned local sources; the Dataset does not import this.

Run: python road_training/tools/prepare_data.py --repository /path/to/pothole
Only NumPy and the Python standard library are needed for preparation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

CHANNELS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "speed"]
LABELS = ["defect", "defect_assumed_normal", "kaggle_type", "synthetic_type", "quality_grade"]
KAGGLE_TYPES = ["manhole", "depression", "bump", "crack"]
KAGGLE_CLASSES = ["normal", *KAGGLE_TYPES]
SYNTHETIC_TYPES = ["pothole", "speed_bump", "crack"]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checksum(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_record(out, info, x, mask, time, labels, iri, metadata):
    """One small folder per continuous recording, suitable for memory mapping."""
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    time = np.asarray(time, dtype=np.float64)
    n = len(time)
    if x.shape != (n, 7) or mask.shape != x.shape or labels.shape != (n, len(LABELS)):
        raise ValueError("Invalid prepared shapes")
    if not np.isfinite(x).all() or np.any(x[~mask] != 0):
        raise ValueError("Inputs must be finite, with masked zero placeholders")
    if not np.allclose(np.diff(time), .01, atol=1e-9, rtol=0):
        raise ValueError("Only a uniform 100 Hz grid is accepted")
    folder = out / "records" / info["id"]
    folder.mkdir(parents=True, exist_ok=False)
    arrays = dict(x=x, mask=mask, time=time, labels=labels.astype(np.int16), iri=iri.astype(np.float32))
    for name, array in arrays.items():
        np.save(folder / f"{name}.npy", array, allow_pickle=False)
    write_json(folder / "metadata.json", metadata)
    return dict(**info, path=folder.relative_to(out).as_posix(), samples=n,
                duration_seconds=n / 100, first_time_s=float(time[0]), last_time_s=float(time[-1]),
                valid_fraction=mask.mean(0).tolist(),
                file_sha256={p.name: checksum(p) for p in sorted(folder.iterdir())})


def kaggle_labels(time, annotations, imu_valid):
    """Keep source taxonomy and distinguish explicit from assumed negatives."""
    labels = np.full((len(time), len(LABELS)), -100, np.int16)
    conflict = np.zeros((len(time), 2), bool)
    touched = np.zeros((len(time), 2), bool)
    # Two independent tasks: overlapping distinct defect types still agree
    # about binary defect presence, but not about the defect's type.
    for a in annotations:
        use = (time >= a["rel_t_start"]) & (time < a["rel_t_end"])
        kind = a["anomaly"]
        values = [1 if kind in KAGGLE_TYPES else 0 if kind == "no_anomaly" else -100,
                  KAGGLE_TYPES.index(kind) if kind in KAGGLE_TYPES else -100]
        for j, (column, value) in enumerate(zip([0, 2], values)):
            conflict[:, j] |= use & touched[:, j] & (labels[:, column] != value)
            labels[use, column] = value
            touched[use, j] = True
    for j, column in enumerate([0, 2]):
        labels[conflict[:, j], column] = -100
    if annotations:
        labels[:, 1] = labels[:, 0]
        labels[~touched[:, 0], 1] = 0
    # Five-class task: retain unknown/conflicting types, add the same explicit
    # weak negatives as defect_assumed_normal, and shift defect IDs to 1..4.
    known_type = labels[:, 2] >= 0
    labels[known_type, 2] += 1
    labels[labels[:, 1] == 0, 2] = 0
    labels[~imu_valid] = -100
    return labels


def prepare_kaggle(repository, out):
    root = repository / "data/sensors"
    protocol_path = repository / "data/prepared/manifest.json"
    protocol = read_json(protocol_path)
    # Reuse the already purged, event-safe intervals, not the old normalized
    # patch tensors. Each real sample belongs to exactly one split.
    entries = {}
    for split in ("train", "val", "test", "pretrain"):
        for entry in protocol["splits"][split]:
            if entry["name"] not in entries:
                entries[entry["name"]] = ("train" if split == "pretrain" else split, entry)
    records = []
    for name, (split, entry) in entries.items():
        drive = name
        if name.startswith("larisa_"):
            drive = name.rsplit("_", 1)[0]
        source = root / f"{drive}.npz"
        with np.load(source, allow_pickle=False) as z:
            lo, hi = entry["first_time"], entry["last_time"]
            use = (z["time"] >= lo) & (z["time"] <= hi)
            native_time = z["time"][use]
            native_imu = z["imu"][use]
            if len(native_time) < 2 or np.any(np.diff(native_time) <= 0):
                raise ValueError(f"Invalid IMU clock in {name}")
            count = int(np.floor((native_time[-1] - native_time[0]) * 100 + 1e-8)) + 1
            time = native_time[0] + np.arange(count) / 100
            # Align jittered ~100 Hz IMU observations onto an exact 100 Hz
            # grid. Interpolation stays entirely within this split.
            x = np.zeros((count, 7), np.float32)
            mask = np.zeros_like(x, bool)
            right = np.clip(np.searchsorted(native_time, time, side="right"), 1, len(native_time)-1)
            gaps = native_time[right] - native_time[right-1]
            for j in range(6):
                x[:, j] = np.interp(time, native_time, native_imu[:, j])
                mask[:, j] = (gaps <= .05) & np.isfinite(native_imu[right, j]) & np.isfinite(native_imu[right-1, j])
            # Speed is last-observed, never interpolated from the future.
            index = np.searchsorted(z["gps_time"], time, side="right") - 1
            safe = np.maximum(index, 0)
            x[:, 6] = z["speed"][safe]
            mask[:, 6] = (index >= 0) & (time - z["gps_time"][safe] <= 3.) & np.isfinite(x[:, 6])
            x[~mask] = 0
        annotations = read_json(root / f"{drive}.labels.json")
        selected = [a for a in annotations if a["rel_t_end"] > lo and a["rel_t_start"] <= hi]
        if any(a["rel_t_start"] < lo or a["rel_t_end"] > hi for a in selected):
            raise ValueError(f"An event crosses the archived split boundary: {name}")
        labels = kaggle_labels(time, selected, mask[:, :6].all(1))
        info = dict(id="kaggle_" + name, source="real", dataset="kaggle", split=split,
                    groups=["kaggle/" + name], original_recording=drive,
                    source_file=str(source.resolve()), source_sha256=checksum(source))
        metadata = dict(source="Kaggle Road Quality Dataset", original_recording=drive,
                        annotations=selected, split=split, archived_interval=entry,
                        split_protocol_sha256=checksum(protocol_path),
                        clock_policy="Uniform 100 Hz interpolation within the archived split; IMU gaps >50 ms masked. Last-observed speed, stale after 3 s.",
                        label_policy="Per-sample intervals, no patch majority vote. Unannotated is unknown in defect; defect_assumed_normal and five-class kaggle_type assume normal outside annotations in annotated drives only. Conflicting/unknown types and invalid IMU remain ignored.",
                        units=["m/s²"] * 3 + ["rad/s"] * 3 + ["m/s"], input_frame="recorded sensor axes; accelerometer includes gravity")
        records.append(write_record(out, info, x, mask, time, labels, np.full(count, np.nan, np.float32), metadata))
        print(f"Prepared {info['id']}: {count:,} samples ({split})", flush=True)
    return records


def prepare_synthetic(repository, out):
    root = repository / "data/crack_frame_learning_v1/corrected_physical"
    sources = [(p, root / "shards" / read_json(p)["file"], "corrected_physical")
               for p in sorted((root / "records").glob("*.json"))]
    if not sources:
        raise ValueError("No corrected physical recordings found")
    gallery = repository / "reports/simulation_gallery"
    # Only the four genuinely new general-road recordings. The other gallery
    # examples repeat corrected source data or a measured calibration response.
    for row in read_json(gallery / "selection.json")["items"]:
        if row["fresh_physics_run"]:
            folder = gallery / row["slug"]
            sources.append((folder / "record.json", folder / "data.npz", "general_road_replay"))
    records, seen, group_splits = [], set(), {}
    for record_path, shard, dataset in sources:
        record = read_json(record_path)
        digest = checksum(shard)
        if digest != record["sha256"] or not record["admission"]["passed"]:
            raise ValueError(f"Unverified synthetic recording: {shard}")
        if dataset == "corrected_physical" and record["observation_correction"]["version"] != "chassis_acceleration_frame_v1":
            raise ValueError("Expected corrected accelerometer frames")
        if digest in seen:
            continue
        seen.add(digest)
        spec = record["scenario"]
        split = {"validation": "val"}.get(spec["dataset_split"], spec["dataset_split"])
        for group in spec["source_group_ids"]:
            if group in group_splits and group_splits[group] != split:
                raise ValueError(f"Synthetic source family crosses splits: {group}")
            group_splits[group] = split
        with np.load(shard, allow_pickle=False) as z:
            x, mask, time = z["X"], z["input_valid"], z["time_s"]
            labels = np.full((len(time), len(LABELS)), -100, np.int16)
            valid = z["defect_valid"]
            labels[valid, 0] = z["defect_context"][valid]
            labels[:, 1] = labels[:, 0]
            # Type is conditional on one unambiguous active defect kind.
            kinds = z["kind_context"] & z["kind_valid"]
            single = kinds.sum(1) == 1
            for j, kind in enumerate(record["kind_order"]):
                labels[single & kinds[:, j], 3] = SYNTHETIC_TYPES.index(kind)
            labels[z["background_valid"], 4] = z["background_class"][z["background_valid"]]
            iri = np.where(z["background_valid"], z["background_iri"], np.nan)
            info = dict(id="synthetic_" + spec["scenario_id"], source="synthetic", dataset=dataset,
                        split=split, groups=["synthetic/" + g for g in spec["source_group_ids"]],
                        original_recording=spec["scenario_id"], source_file=str(shard.resolve()), source_sha256=digest)
            row = write_record(out, info, x, mask, time, labels, iri, record)
            # Preserve original timing, event IDs, localization/support targets
            # and per-task masks for later specialized fine-tuning scripts.
            target_path = out / row["path"] / "original_targets.npz"
            np.savez_compressed(target_path, **{k: z[k] for k in z.files if k not in ("X", "input_valid")})
            row["file_sha256"][target_path.name] = checksum(target_path)
            # Synthetic sensor values/masks/clocks must remain exact copies.
            for key, original in [("x", x), ("mask", mask), ("time", time)]:
                np.testing.assert_array_equal(np.load(out / row["path"] / f"{key}.npy"), original)
            records.append(row)
    return records


def training_statistics(out, records, source, *, allow_missing_channels=False):
    """Compute per-channel moments from available TRAIN observations only."""
    count = np.zeros(7, np.int64)
    total = np.zeros(7, np.float64)
    squares = np.zeros(7, np.float64)
    ids = []
    for record in records:
        if record["split"] != "train" or (source != "both" and record["source"] != source):
            continue
        folder = out / record["path"]
        x = np.load(folder / "x.npy", mmap_mode="r")
        mask = np.load(folder / "mask.npy", mmap_mode="r")
        ids.append(record["id"])
        for start in range(0, len(x), 65536):
            values = np.where(mask[start:start+65536], x[start:start+65536], 0).astype(np.float64)
            count += mask[start:start+65536].sum(0)
            total += values.sum(0)
            squares += np.square(values).sum(0)
    if np.any(count == 0) and not allow_missing_channels:
        raise ValueError(f"No training observations for a channel in {source}")
    mean = total / np.maximum(count, 1)
    std = np.sqrt(np.maximum(squares / np.maximum(count, 1) - mean**2, 1e-12))
    std[count == 0] = 1.  # A fully missing LiRA gyroscope is masked, never fabricated.
    return dict(mean=mean.tolist(), std=std.tolist(), count=count.tolist(), recording_ids=ids,
                fit="Available TRAIN samples only; no val/test values, no per-window or per-recording scaling.")


def build(repository, out):
    repository, out = Path(repository).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = prepare_kaggle(repository, out) + prepare_synthetic(repository, out)
    counts = Counter((r["source"], r["split"]) for r in records)
    summary = {source: {split: dict(recordings=counts[source, split],
                                    seconds=sum(r["duration_seconds"] for r in records if r["source"] == source and r["split"] == split))
                        for split in ["train", "val", "test"]} for source in ["real", "synthetic"]}
    manifest = dict(version=2, sample_rate_hz=100, channels=CHANNELS,
                    units=["m/s²"]*3 + ["rad/s"]*3 + ["m/s"], label_columns=LABELS,
                    label_classes=dict(kaggle_type=KAGGLE_CLASSES, synthetic_type=SYNTHETIC_TYPES,
                                       quality_grade=["good", "medium", "bad", "terrible"]),
                    unknown_class=-100, records=records, summary=summary,
                    train_statistics={source: training_statistics(out, records, source) for source in ["real", "synthetic", "both"]},
                    preparation_script_sha256=checksum(__file__),
                    sources_policy="Kaggle raw IMU/speed inside existing purged splits; 63 corrected physical synthetic records and four fresh admitted general-road replays. No alias/repeat views, old buggy observer exports, pending steering experiments or gallery duplicates.",
                    excluded_sources="This base build contains Kaggle and synthetic data. Run tools/prepare_lira.py to add LiRA accelerometer/speed and measured section IRI with missing gyros masked. Measured-road calibration replays remain excluded.",
                    synthetic_test_policy="This corrected corpus has TRAIN/VAL only. No synthetic test split is invented or borrowed from VAL.")
    write_json(out / "manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    build(args.repository, args.output)
