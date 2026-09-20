"""Add verified RoadSens-4M IMU recordings to a NEW prepared dataset.

The published combined CSVs already have a 100 Hz grid. We retain total
acceleration (including gravity) and calibrated gyro in recorded device axes.
There is no vehicle speed or measured IRI in these files. RoadSens is TRAIN
only; the base dataset's held-out roads and labels are preserved exactly.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from road_training.dataset import collection_name
from road_training.tools.prepare_data import checksum, training_statistics, write_json, write_record

ARTICLE_API = "https://api.figshare.com/v2/articles/30341143/versions/3"
DATASET_DOI = "https://doi.org/10.6084/m9.figshare.30341143.v3"
COMBINED_FILE = "Combined CSV with GIS and Weather Data.zip"
COMBINED_MD5 = "371536cc260e2a590c972ce0b1fe43bd"
COMBINED_URL = "https://ndownloader.figshare.com/files/58678312"
ISOLATED_MD5 = "4d0d3a60f08e431019e8130ab43f1fdb"
INPUT_COLUMNS = [f"{sensor}_{axis}" for sensor in ("totalAcceleration", "gyroscope") for axis in "xyz"]
TYPE_NAMES = ("normal", "bump", "pothole")


def acquire(source, outer_archive=None):
    """Verify the pinned release, extract only its continuous sensor CSVs.

    Isolated event subsets repeat samples and destroy continuous time context.
    One complete session (99) is misfiled inside an isolated archive; recover
    only its continuous combined CSV. GIS and videos are not extracted.
    bsdtar (libarchive) is needed only to read the two nested RAR files.
    """
    source = Path(source)
    source.mkdir(parents=True, exist_ok=True)
    if not shutil.which("bsdtar"):
        raise RuntimeError("Install bsdtar/libarchive to read the publisher's nested RAR files")
    with urllib.request.urlopen(ARTICLE_API, timeout=60) as response:
        article = json.load(response)
    entry = next(f for f in article["files"] if f["name"] == COMBINED_FILE)
    if entry["computed_md5"] != COMBINED_MD5:
        raise ValueError("Pinned Figshare v3 file changed")
    if outer_archive is not None and Path(outer_archive).exists():
        with zipfile.ZipFile(outer_archive) as archive:
            payload = archive.read(COMBINED_FILE)
        origin = dict(local_archive=str(Path(outer_archive).resolve()), sha256=checksum(outer_archive))
    else:
        cached = source / COMBINED_FILE
        if not cached.exists():
            pending = source / "combined.download"
            with urllib.request.urlopen(COMBINED_URL, timeout=120) as response, pending.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if hashlib.md5(pending.read_bytes()).hexdigest() != COMBINED_MD5:
                raise ValueError("Download checksum mismatch")
            pending.rename(cached)
        payload = cached.read_bytes()
        origin = dict(download_url=COMBINED_URL)
    if len(payload) != entry["size"] or hashlib.md5(payload).hexdigest() != COMBINED_MD5:
        raise ValueError("Combined archive does not match Figshare v3")
    inputs = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member, category in [
            ("Combined CSV with GIS and Weather Data/Normal Road (No Annotation)/Normal Road (No Annotation).rar", "normal"),
            ("Combined CSV with GIS and Weather Data/Road Anomalies/Road Anomalies.rar", "anomalies"),
        ]:
            rar = source / Path(member).name
            data = archive.read(member)
            if rar.exists() and rar.read_bytes() != data:
                raise ValueError(f"Existing source differs from the release: {rar}")
            if not rar.exists():
                rar.write_bytes(data)
            names = subprocess.check_output(["bsdtar", "-tf", str(rar)], text=True).splitlines()
            if not all(Path(n).name == n and n.endswith(".csv") for n in names):
                raise ValueError("Unexpected archive layout; refusing unscoped extraction")
            destination = source / "combined" / category
            destination.mkdir(parents=True, exist_ok=True)
            for name in names:
                raw = subprocess.check_output(["bsdtar", "-xOf", str(rar), name])
                target = destination / name
                if target.exists() and target.read_bytes() != raw:
                    raise ValueError(f"Existing CSV differs from release: {target}")
                if not target.exists():
                    target.write_bytes(raw)
                inputs.append(dict(path=str(target.resolve()), category=category, sha256=checksum(target)))
    # In the combined archive, 99.csv is accidentally a summary table. Its
    # full, continuous sensor recording lives in the second archive instead.
    isolated = next(f for f in article["files"] if f["name"] == "Isolated Data.zip")
    if isolated["computed_md5"] != ISOLATED_MD5:
        raise ValueError("Pinned isolated archive changed")
    if outer_archive is not None and Path(outer_archive).exists():
        with zipfile.ZipFile(outer_archive) as archive:
            recovered_payload = archive.read("Isolated Data.zip")
    else:
        cached = source / "Isolated Data.zip"
        if not cached.exists():
            pending = source / "isolated.download"
            with urllib.request.urlopen(isolated["download_url"], timeout=120) as response, pending.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if hashlib.md5(pending.read_bytes()).hexdigest() != ISOLATED_MD5:
                raise ValueError("Isolated archive download checksum mismatch")
            pending.rename(cached)
        recovered_payload = cached.read_bytes()
    if len(recovered_payload) != isolated["size"] or hashlib.md5(recovered_payload).hexdigest() != ISOLATED_MD5:
        raise ValueError("Isolated archive does not match Figshare v3")
    with zipfile.ZipFile(io.BytesIO(recovered_payload)) as archive:
        rar_data = archive.read("Isolated Data/Road Anomalies/87-103-GIS-integrated.rar")
    rar = source / "87-103-GIS-integrated.rar"
    if rar.exists() and rar.read_bytes() != rar_data:
        raise ValueError("Existing recovery RAR differs from the release")
    if not rar.exists():
        rar.write_bytes(rar_data)
    member = "99-isolated-anomalies/99_combined_with_annotations_and_gis.csv"
    recovered_csv = subprocess.check_output(["bsdtar", "-xOf", str(rar), member])
    target = source / "combined" / "recovered" / "99.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != recovered_csv:
        raise ValueError("Existing recovery CSV differs from the release")
    if not target.exists():
        target.write_bytes(recovered_csv)
    inputs.append(dict(path=str(target.resolve()), category="anomalies", sha256=checksum(target),
                       recovered_from=member, archive="Isolated Data.zip"))
    protocol = dict(dataset="RoadSens-4M", doi=DATASET_DOI, article_api=ARTICLE_API,
                    license=article["license"], authors=article.get("authors", []),
                    archive=entry, recovery_archive=isolated, origin=origin, inputs=inputs,
                    excluded="GIS archive, duplicate isolated subsets, videos, non-IMU model inputs")
    write_json(source / "source_protocol.json", protocol)
    return protocol


def construct_labels(table, normal_recording=False):
    """Only explicit normal recordings and bounded two-button events supervise.

    The annotation text is already expanded between button presses. Press
    duration describes the button, not the road event's duration. Blank text
    in an anomaly recording is unknown, not a verified normal-road target.
    A one-button trailing interval is also unknown. No signal-derived labels.
    """
    n = len(table)
    labels = np.full((n, 5), -100, np.int16)
    types = np.full(n, -100, np.int16)
    events = []
    text = table.get("annotation_text", pd.Series("", index=table.index)).fillna("").str.strip().str.lower().to_numpy()
    if normal_recording:
        if np.any(text != ""):
            raise ValueError("Normal-only recording contains event annotations")
        labels[:, :2] = 0
        types[:] = 0
        return labels, types, events
    buttons = pd.to_numeric(table["annotation_millisecond_press_duration"], errors="raise").to_numpy()
    edges = np.r_[0, np.flatnonzero(text[1:] != text[:-1]) + 1, n]
    for start, stop in zip(edges[:-1], edges[1:]):
        kind = text[start]
        if not kind:
            continue
        presses = np.flatnonzero(np.isfinite(buttons[start:stop]))
        closed = (kind in TYPE_NAMES[1:] and len(presses) == 2
                  and presses[0] == 0 and presses[-1] == stop-start-1
                  and np.all(buttons[start:stop][presses] > 0))
        if closed:
            labels[start:stop, :2] = 1
            types[start:stop] = TYPE_NAMES.index(kind)
        events.append(dict(kind=str(kind), first_sample=int(start), stop_sample=int(stop),
                           duration_seconds=(stop-start)/100, admitted=bool(closed),
                           button_count=len(presses)))
    return labels, types, events


def convert(table, normal_recording=False):
    """Preserve each 10 ms observation and mask missing values without filling."""
    time = table["seconds_elapsed"].to_numpy(dtype=np.float64)
    if not len(time) or not np.isfinite(time).all() or not np.allclose(np.diff(time), .01, rtol=0, atol=1e-8):
        raise ValueError("Expected a continuous, ordered 100 Hz recording; never concatenate isolated events")
    values = table[INPUT_COLUMNS].to_numpy(dtype=np.float32)
    mask = np.zeros((len(time), 7), bool)
    mask[:, :6] = np.isfinite(values)
    x = np.zeros((len(time), 7), np.float32)
    x[:, :6] = np.where(mask[:, :6], values, 0)
    labels, types, events = construct_labels(table, normal_recording)
    valid = mask[:, :6].all(1)
    labels[~valid] = -100
    types[~valid] = -100
    return x, mask, time, labels, types, events


def prepare(base, source, out, protocol):
    """Build separately so prior experiments keep their data and statistics."""
    base, source, out = Path(base).resolve(), Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new dataset destination: {out}")
    before = (base / "manifest.json").read_bytes()
    manifest = json.loads(before)
    if any(collection_name(r) == "roadsens" for r in manifest["records"]):
        raise ValueError("Base already includes RoadSens")
    audit = dict(excluded=[], duplicates=[], records=[], policy="RoadSens TRAIN only; existing VAL/TEST unchanged")
    prepared, signatures = [], {}
    for item in sorted(protocol["inputs"], key=lambda r: int(Path(r["path"]).stem.split()[0])):
        path = Path(item["path"])
        if checksum(path) != item["sha256"]:
            raise ValueError(f"Changed source CSV: {path}")
        table = pd.read_csv(path)
        if "seconds_elapsed" not in table:
            audit["excluded"].append(dict(file=path.name, reason="summary table, no sensor time series"))
            continue
        x, mask, time, labels, types, events = convert(table, item["category"] == "normal")
        # Ignore unrelated GIS/weather columns when detecting copied recordings.
        digest = hashlib.sha256()
        for value in (x, mask, np.rint((time-time[0])*100).astype(np.int64)):
            digest.update(np.ascontiguousarray(value).tobytes())
        signature = digest.hexdigest()
        label_signature = hashlib.sha256(labels.tobytes() + types.tobytes()).hexdigest()
        if signature in signatures:
            original, original_labels = signatures[signature]
            if label_signature != original_labels:
                raise ValueError(f"Identical IMU has contradictory labels: {original}, {path}")
            audit["duplicates"].append(dict(file=path.name, kept=original, signal_sha256=signature))
            continue
        signatures[signature] = (path.name, label_signature)
        prepared.append((item, x, mask, time, labels, types, events, signature))
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".roadsens_prepare_", dir=out.parent) as tmp:
        staging = Path(tmp) / "dataset"
        (staging / "records").mkdir(parents=True)
        # Link frozen base records; never rewrite them or the old manifest.
        for record in manifest["records"]:
            target = staging / "records" / record["id"]
            target.symlink_to((base / record["path"]).resolve(), target_is_directory=True)
            record["path"] = target.relative_to(staging).as_posix()
        new = []
        for item, x, mask, time, labels, types, events, signature in prepared:
            path = Path(item["path"])
            session = int(path.stem.split()[0])
            info = dict(id=f"roadsens_{session:03d}", source="real", dataset="roadsens4m", split="train",
                        groups=[f"roadsens/session_{session:03d}"], original_recording=path.stem,
                        source_file=str(path), source_sha256=item["sha256"], signal_sha256=signature)
            metadata = dict(source="RoadSens-4M", doi=DATASET_DOI, source_file=str(path),
                            source_sha256=item["sha256"], split="train", events=events,
                            input_columns=INPUT_COLUMNS, units=["m/s²"]*3+["rad/s"]*3+["m/s"],
                            input_frame="Recorded device axes; total acceleration includes gravity. No inferred vehicle-axis rotation.",
                            clock_policy="Published 100 Hz combined grid copied without interpolation or filling; upstream aggregation is publisher-defined.",
                            label_policy="Only two-button closed bump/pothole intervals and explicit normal-only files; all other samples unknown. No Kaggle type or IRI mapping.",
                            missing_channels=["speed"], roadsens_type_classes=TYPE_NAMES,
                            supervision="Coarse manual event intervals, not exact tire contact or pointwise road severity")
            record = write_record(staging, info, x, mask, time, labels, np.full(len(time), np.nan, np.float32), metadata)
            type_path = staging / record["path"] / "roadsens_type.npy"
            np.save(type_path, types, allow_pickle=False)
            record["file_sha256"][type_path.name] = checksum(type_path)
            new.append(record)
            counts = Counter(map(int, labels[:, 1]))
            audit["records"].append(dict(id=info["id"], samples=len(time), seconds=len(time)/100,
                normal_samples=counts[0], disturbance_samples=counts[1], unknown_samples=counts[-100],
                bump_samples=int((types == 1).sum()), pothole_samples=int((types == 2).sum()),
                missing_imu_rows=int((~mask[:, :6].all(1)).sum()),
                closed_events=sum(e["admitted"] for e in events), ambiguous_events=sum(not e["admitted"] for e in events),
                windows_stride_1=max(0, len(time)-1024+1)))
        manifest["records"] += new
        manifest["train_statistics"] = {s: training_statistics(staging, manifest["records"], s, allow_missing_channels=True)
                                        for s in ("real", "synthetic", "both")}
        manifest["train_statistics_by_real_dataset"] = {}
        for name in ("kaggle", "lira", "roadsens"):
            chosen = [r for r in manifest["records"] if r["source"] != "real" or collection_name(r) == name]
            manifest["train_statistics_by_real_dataset"][name] = {
                s: training_statistics(staging, chosen, s, allow_missing_channels=True) for s in ("real", "synthetic", "both")}
        train_ids = {r["id"] for r in manifest["records"] if r["split"] == "train"}
        for stats in [*manifest["train_statistics"].values(),
                      *[v for d in manifest["train_statistics_by_real_dataset"].values() for v in d.values()]]:
            if not set(stats["recording_ids"]) <= train_ids:
                raise ValueError("Held-out record used for normalization")
        audit["totals"] = {key: sum(r[key] for r in audit["records"]) for key in audit["records"][0] if key != "id"}
        audit["totals"]["recordings"] = len(new)
        manifest["roadsens4m_integration"] = dict(
            version=1, base_root=str(base), base_manifest_sha256=hashlib.sha256(before).hexdigest(),
            recipe_sha256=checksum(__file__), source_protocol_sha256=checksum(source / "source_protocol.json"),
            doi=DATASET_DOI, license=protocol["license"], split_policy=audit["policy"],
            label_policy="Explicit normal recordings and closed bump/pothole intervals only; blank and unclosed annotations unknown",
            unavailable_targets=["IRI", "ordinal road quality", "Kaggle defect taxonomy"],
            unavailable_inputs=["speed"], **{k: audit[k] for k in ("totals", "excluded", "duplicates")})
        manifest["summary"] = {s: {split: dict(
            recordings=sum(r["source"] == s and r["split"] == split for r in manifest["records"]),
            seconds=sum(r["duration_seconds"] for r in manifest["records"] if r["source"] == s and r["split"] == split))
            for split in ("train", "val", "test")} for s in ("real", "synthetic")}
        manifest["sources_policy"] += " RoadSens-4M IMU only, deduplicated TRAIN addition; explicit known labels only."
        (staging / "manifest.before_roadsens.json").write_bytes(before)
        write_json(staging / "roadsens_audit.json", audit)
        write_json(staging / "manifest.json", manifest)
        if (base / "manifest.json").read_bytes() != before:
            raise ValueError("Base manifest changed during preparation")
        staging.rename(out)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=Path("road_training/data_transfer_v2_mount"))
    parser.add_argument("--source-root", type=Path, default=Path("data/roadsens4m"))
    parser.add_argument("--archive", type=Path, default=Path("30341143.zip"))
    parser.add_argument("--output", type=Path, default=Path("road_training/data_with_roadsens"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new destination")
    protocol = acquire(args.source_root, args.archive)
    audit = prepare(args.base_root, args.source_root, args.output, protocol)
    print(json.dumps(audit["totals"], indent=2))


if __name__ == "__main__":
    main()
