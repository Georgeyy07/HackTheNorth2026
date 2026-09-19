"""Export complete test drives for timestamp-correct replay on a map.

Run from the repository: python -m road_training.export_test_drives
The trained ensemble and Timeline are reused without fitting or threshold tuning.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
import torch

from road_training.checkpoints import load_teachers, model_channels, channel_names
from road_training.acceleration_speed import INDICES
from road_training.timeline_stream import RoadTimelineStream


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "artifacts/data_roadsens_aligned"
RECEIPT = REPO / "models/acceleration_speed/ensemble.json"
SELECTION = REPO / "configs/timeline.json"
OUTPUT = REPO / "artifacts/test_drive_inference"
CHANNELS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "speed"]
GRADES = ["good", "medium", "bad", "terrible"]
HZ, PATCH = 100, 16


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def clean_observations(values):
    """Keep the last duplicate timestamp, preserving chronological order."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values).all(1)]
    values = values[np.argsort(values[:, 0], kind="stable")]
    if not len(values):
        return values
    return values[np.r_[np.diff(values[:, 0]) > 0, True]]


def causal_gps(query, fixes, max_age_s=3.):
    """GPS columns are session seconds, latitude, longitude. Never look ahead."""
    query = np.asarray(query, dtype=float)
    n = len(query)
    if not len(fixes):
        return dict(latitude_deg=np.full(n, np.nan), longitude_deg=np.full(n, np.nan),
                    gps_valid=np.zeros(n, bool), gps_age_s=np.full(n, np.nan),
                    gps_fix_index=np.full(n, -1, np.int64))
    if np.any(np.diff(fixes[:, 0]) <= 0):
        raise ValueError("GPS fixes must be unique and increasing")
    index = np.searchsorted(fixes[:, 0], query, side="right") - 1
    safe = np.maximum(index, 0)
    age = np.where(index >= 0, query - fixes[safe, 0], np.nan)
    valid = (index >= 0) & (age >= 0) & (age <= max_age_s)
    return dict(latitude_deg=np.where(valid, fixes[safe, 1], np.nan),
                longitude_deg=np.where(valid, fixes[safe, 2], np.nan),
                gps_valid=valid, gps_age_s=age, gps_fix_index=index)


def interpolation_ready(grid, native_time):
    """When both endpoints used by np.interp have actually been observed.

    Kaggle's existing preparation interpolates within TEST. Account for that
    short input delay instead of pretending the interpolated values were live.
    Exact native timestamps require only that observation.
    """
    index = np.searchsorted(native_time, grid, side="left")
    if len(native_time) < 2 or np.any(index >= len(native_time)) or grid[0] < native_time[0]:
        raise ValueError("Prepared grid exceeds native observations")
    return native_time[index]


def native_sources(record, folder, source_time):
    """Load original sensor/GPS observations with clocks relative to TEST start.

    No preceding model context is used to initialize a drive. The existing
    prepared speed may hold an observation just before the first IMU sample;
    its original source timestamp is retained separately.
    LiRA gps is [UNIX,lat,lon]; gps_mapmatch has the opposite coordinate order
    and is deliberately not used as the measured vehicle position.
    """
    metadata = read(folder / "metadata.json")
    t0, last = float(source_time[0]), float(source_time[-1])
    if record["dataset"] == "kaggle":
        with np.load(record["source_file"], allow_pickle=False) as z:
            interval = metadata["archived_interval"]
            keep = (z["time"] >= interval["first_time"]) & (z["time"] <= interval["last_time"])
            native_time, imu = z["time"][keep].copy(), z["imu"][keep].copy()
            ready = interpolation_ready(source_time, native_time) - t0
            imu_rows = np.column_stack((native_time - t0, imu))
            end = interval["last_time"]
            keep = (z["gps_time"] >= t0) & (z["gps_time"] <= end)
            fixes = np.column_stack((z["gps_time"][keep] - t0, z["lat"][keep], z["lon"][keep]))
            speed = np.column_stack((z["gps_time"][keep] - t0, z["speed"][keep]))
            # Model speed may use an earlier observed fix. Preserve provenance
            # in its ready clock, without importing GPS from a previous split.
            speed_index = np.searchsorted(z["gps_time"], source_time, side="right") - 1
            speed_source = np.where(speed_index >= 0, z["gps_time"][np.maximum(speed_index, 0)] - t0, np.nan)
            origin_ns = int(z["absolute_start"].item()) + int(round(t0 * 1e9))
        imu_columns = CHANNELS[:6]
        policy = "Existing TEST-only linear IMU interpolation to 100 Hz; measured SI values before model normalization. Original irregular observations are in imu_native.parquet."
        acc_source = np.full(len(source_time), np.nan)  # interpolation has two endpoints
        gps_source = "Kaggle NPZ gps_time, lat, lon; measured GPS, no map matching"
    elif record["dataset"] == "lira_cd":
        origin = float(metadata["unix_time_origin_s"])
        with h5py.File(record["source_file"], "r") as handle:
            group = handle[metadata["pass_name"]]
            names = list(group["gps"].attrs["chNames"])
            if names[1:] != ["lat", "lon"]:
                raise ValueError(f"Unexpected LiRA GPS coordinate schema: {names}")
            acc = clean_observations(group["acc.xyz"][...])
            speed = clean_observations(group[metadata["speed_key"]][...])
            fixes = clean_observations(group["gps"][...])
        for values in (acc, speed, fixes):
            values[:, 0] -= origin + t0
        imu_rows = acc[(acc[:, 0] >= 0) & (acc[:, 0] < last - t0 + .01)].copy()
        imu_rows[:, 1:] *= 9.80665
        speed = speed[(speed[:, 0] >= 0) & (speed[:, 0] < last - t0 + .01)].copy()
        speed[:, 1:] /= 3.6
        fixes = fixes[(fixes[:, 0] >= 0) & (fixes[:, 0] < last - t0 + .01)]
        sensor_time = np.load(folder / "sensor_source_time.npy") - t0
        acc_source, speed_source = sensor_time.T
        ready = np.maximum(source_time - t0, np.nan_to_num(sensor_time, nan=-np.inf).max(1))
        origin_ns = int(round(origin * 1e9)) + int(round(t0 * 1e9))
        imu_columns = CHANNELS[:3]
        policy = "Existing causal last-observed acc.xyz and speed on 100 Hz grid. Original acceleration converted g to m/s^2; speed km/h to m/s. Gyroscope unavailable, never synthesized."
        gps_source = f"LiRA HDF5 {metadata['pass_name']}/gps [UNIX seconds, latitude, longitude]; no gps_mapmatch or reference-survey coordinates"
    else:
        raise ValueError("Only Kaggle and LiRA test data are supported")
    fixes = clean_observations(fixes)
    fixes = fixes[(np.abs(fixes[:, 1]) <= 90) & (np.abs(fixes[:, 2]) <= 180)]
    return dict(imu=imu_rows, imu_columns=imu_columns, speed=clean_observations(speed),
                fixes=fixes, origin_ns=origin_ns, input_ready_s=ready,
                accel_source_time_s=acc_source, speed_source_time_s=speed_source,
                input_policy=policy, gps_source=gps_source)


def available_time(emitted_samples, input_ready):
    """Data availability excluding compute, respecting interpolation endpoints."""
    return max(emitted_samples / HZ, float(np.max(input_ready[:emitted_samples], initial=0.)))


@torch.inference_mode()
def infer_drive(model, x, mask, config, on_update=None):
    """Exact B=1 live adapter; collect final and evolving provisional estimates.

    The partial last patch gets one separately marked EOF estimate with masked
    unobserved samples. It never enters Timeline or finalizes its earlier tail.
    """
    stream = RoadTimelineStream(model, config)
    x = torch.as_tensor(x, dtype=torch.float32, device=stream.device)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=stream.device)
    if x.ndim != 2 or mask.shape != x.shape:
        raise ValueError('Input and mask must be matching two-dimensional arrays')
    # Prepared files retain their canonical raw sensor columns for provenance.
    # Remove gyro before windowing, normalization, validity or model inference.
    if x.ndim == 2 and x.shape[1] == 7 and stream.channels == 4:
        x, mask = x[:, list(INDICES)], mask[:, list(INDICES)]
    if x.ndim != 2 or x.shape[1] != stream.channels or mask.shape != x.shape:
        raise ValueError('Input and mask must match the model or canonical recording schema')
    if not torch.isfinite(x[mask]).all():
        raise ValueError('Observed inputs must be finite, including the partial final patch')
    n, rows = len(x), []
    whole = n // PATCH * PATCH
    for start in range(0, whole, PATCH):
        final = stream.push(x[start:start + PATCH], mask[start:start + PATCH])
        rows.extend(final)
        if on_update:
            for row in final:
                on_update(dict(row))
            for provisional in stream.provisional():
                target = provisional["target_patch"]
                on_update(dict(provisional, target_sample_start=target * PATCH,
                               target_sample_end=(target + 1) * PATCH,
                               emitted_after_samples=start + PATCH,
                               valid=provisional["votes"] > 0))
    for provisional in stream.provisional():
        target = provisional["target_patch"]
        rows.append(dict(provisional, target_sample_start=target * PATCH,
                         target_sample_end=(target + 1) * PATCH,
                         emitted_after_samples=whole, valid=provisional["votes"] > 0))
    if n > whole:
        tail = torch.zeros(1, PATCH, stream.channels, device=stream.device)
        tail_mask = torch.zeros_like(tail, dtype=torch.bool)
        tail[0, :n - whole] = x[whole:]
        tail_mask[0, :n - whole] = mask[whole:]
        window = torch.cat((stream.window[:, PATCH:], tail), 1)
        observed = torch.cat((stream.window_mask[:, PATCH:], tail_mask), 1)
        with torch.autocast(stream.device.type, dtype=torch.bfloat16, enabled=stream.device.type == "cuda"):
            output = model(window, observed)
        valid = bool(output["patch_valid"][0, -1])
        row = dict(target_patch=whole // PATCH, target_sample_start=whole,
                   target_sample_end=n, emitted_after_samples=n,
                   status="provisional_partial", valid=valid, votes=int(valid),
                   probability=float(output["disturbance_logit"][0, -1].float().sigmoid()) if valid else None,
                   iri_raw_m_per_km=float(output["roughness"][0, -1].float()) if valid else None)
        row = stream.postprocess(row)
        rows.append(row)
        if on_update:
            on_update(dict(row))
    finish = dict(stream.finish(), incomplete_samples=n - whole, total_samples=n)
    return rows, finish


def enrich(row, session_id, input_ready, origin_ns, source_start, fixes):
    """Add replay clocks, patch GPS, and explicit provisional output semantics."""
    row = dict(row)
    start, end = row["target_sample_start"], row["target_sample_end"]
    emitted = row["emitted_after_samples"]
    row.update(session_id=session_id, start_s=start / HZ, end_s=end / HZ,
               available_s=available_time(emitted, input_ready),
               observed_context_start_s=max(0, ((emitted + PATCH - 1) // PATCH) * PATCH - 1024) / HZ,
               observed_context_end_s=emitted / HZ,
               observed_target_samples=end - start)
    row["source_start_s"] = source_start + row["start_s"]
    row["timestamp_unix_ms"] = origin_ns / 1e6 + row["start_s"] * 1000
    row["available_unix_ms"] = origin_ns / 1e6 + row["available_s"] * 1000
    row["is_final"] = row["status"] == "final"
    if not row["is_final"]:
        row.update(disturbance=None, event_id=None, event_transition=None,
                   iri_m_per_km=row.get("iri_raw_m_per_km"), context_spread=None)
        iri = row["iri_m_per_km"]
        row["quality_grade"] = int(np.digitize(iri, [2., 4., 6.])) if iri is not None else None
    row["quality_name"] = GRADES[row["quality_grade"]] if row["quality_grade"] is not None else None
    # The midpoint describes the target road segment, not the car at alert time.
    row["gps_target_time_s"] = (start + end - 1) / (2 * HZ)
    for prefix, query in (("target_", row["gps_target_time_s"]), ("emission_", row["available_s"])):
        location = causal_gps([query], fixes)
        for key, value in location.items():
            item = value[0].item()
            row[prefix + key] = None if isinstance(item, float) and not np.isfinite(item) else item
    return row


def sample_table(x, mask, source_time, native, patches):
    """One row per prepared sample; each 160-ms estimate is explicitly repeated."""
    n = len(x)
    frame = pd.DataFrame(dict(sample_index=np.arange(n), time_s=np.arange(n) / HZ,
                              source_time_s=source_time,
                              timestamp_unix_ms=native["origin_ns"] / 1e6 + np.arange(n) * 10.,
                              input_available_s=native["input_ready_s"],
                              accel_source_time_s=native["accel_source_time_s"],
                              speed_source_time_s=native["speed_source_time_s"]))
    for j, name in enumerate(CHANNELS):
        frame[name] = np.where(mask[:, j], x[:, j], np.nan)
        frame[name + "_valid"] = mask[:, j]
    for name, values in causal_gps(frame.time_s.to_numpy(), native["fixes"]).items():
        frame[name] = values
    columns = ["target_patch", "status", "is_final", "valid", "available_s", "available_unix_ms",
               "probability", "disturbance", "iri_m_per_km", "quality_grade", "quality_name", "votes", "context_spread"]
    predictions = pd.DataFrame(patches).set_index("target_patch", drop=False)
    columns += [name for name in ("original_probability", "score_kind") if name in predictions]
    expanded = predictions.loc[np.arange(n) // PATCH, columns].reset_index(drop=True)
    expanded = expanded.rename(columns={"valid": "prediction_valid", "available_s": "prediction_available_s",
                                        "available_unix_ms": "prediction_available_unix_ms"})
    frame = pd.concat((frame, expanded), axis=1)
    # A closing patch carries event_transition=end, but is NOT inside the event.
    active_ids = [r.get("event_id") if r.get("disturbance") else None for r in patches]
    frame["event_id"] = pd.array(np.asarray(active_ids, object)[np.arange(n) // PATCH], dtype="Int64")
    frame["disturbance"] = pd.array(frame.disturbance, dtype="boolean")
    frame["quality_grade"] = pd.array(frame.quality_grade, dtype="Int64")
    return frame


def event_table(patches, finish):
    """Summarize committed events; censor at the last classified target boundary."""
    events, active = [], None
    for row in patches:
        if not row["is_final"]:
            continue
        transition = row.get("event_transition")
        if transition == "start":
            active = dict(session_id=row["session_id"], event_id=row["event_id"],
                          start_s=row["start_s"], alert_available_s=row["available_s"],
                          alert_available_unix_ms=row["available_unix_ms"],
                          start_unix_ms=row["timestamp_unix_ms"],
                          latitude_deg=row["target_latitude_deg"], longitude_deg=row["target_longitude_deg"],
                          gps_valid=row["target_gps_valid"], gps_age_s=row["target_gps_age_s"],
                          peak_probability=0., classified_until_s=row["end_s"], patches=0)
        if row.get("disturbance"):
            if active is None:
                raise AssertionError("Active event without a start transition")
            active["peak_probability"] = max(active["peak_probability"], row["probability"])
            active["classified_until_s"] = row["end_s"]
            active["patches"] += 1
        if transition in ("end", "censored"):
            if active is None:
                raise AssertionError("End transition without an active event")
            active.update(end_s=row["start_s"] if transition == "end" else None,
                          end_censored=transition == "censored",
                          censor_reason="missing_observations" if transition == "censored" else None,
                          closure_available_s=row["available_s"])
            events.append(active)
            active = None
    if active:
        active.update(end_s=None, end_censored=True, censor_reason="unfinalized_recording_tail",
                      closure_available_s=finish["total_samples"] / HZ)
        events.append(active)
    return events


def write_frame(path, frame, csv=True):
    frame.to_parquet(path.with_suffix(".parquet"), index=False, compression="zstd")
    if csv:
        frame.to_csv(path.with_suffix(".csv.gz"), index=False, compression="gzip")


def write_native(out, native, source_start):
    for name, values, columns in (("imu_native", native["imu"], native["imu_columns"]),
                                  ("speed_native", native["speed"], ["speed"]),
                                  ("gps_fixes", native["fixes"], ["latitude_deg", "longitude_deg"])):
        frame = pd.DataFrame(values, columns=["time_s", *columns])
        frame["source_time_s"] = frame.time_s + source_start
        frame["timestamp_unix_ms"] = native["origin_ns"] / 1e6 + frame.time_s * 1000
        if name == "gps_fixes":
            frame.insert(0, "gps_fix_index", np.arange(len(frame)))
        write_frame(out / name, frame, csv=name == "gps_fixes")


def write_ground_truth(out, folder):
    """Sidecar only: never passed to the model, GPS alignment, or Timeline."""
    labels = np.load(folder / "labels.npy")
    frame = pd.DataFrame(dict(sample_index=np.arange(len(labels))))
    for j, name in enumerate(("defect_explicit", "defect_assumed_normal", "kaggle_type", "synthetic_type", "quality_grade")):
        frame[name] = pd.array(np.where(labels[:, j] >= 0, labels[:, j], np.nan), dtype="Int64")
    if (folder / "overall_iri.npy").exists():
        frame["overall_iri_m_per_km"] = np.load(folder / "overall_iri.npy")
        frame["roughness_section"] = np.load(folder / "roughness_section.npy")
    frame.to_parquet(out / "ground_truth.parquet", index=False, compression="zstd")
    metadata = read(folder / "metadata.json")
    write_json(out / "reference_annotations.json", dict(annotations=metadata.get("annotations", []),
               sections=metadata.get("sections", []), label_policy=metadata.get("label_policy", metadata.get("target_definition"))))


def geojson(out, fixes, events):
    cuts = np.r_[0, np.flatnonzero(np.diff(fixes[:, 0]) > 3.) + 1, len(fixes)]
    segments = [fixes[a:b, [2, 1]].tolist() for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= 2]
    write_json(out / "route.geojson", dict(type="FeatureCollection", features=[dict(type="Feature",
        properties=dict(source="measured GPS; gaps over 3 seconds split the route", replay_requires="gps_fixes timestamps"),
        geometry=dict(type="MultiLineString", coordinates=segments))]))
    write_json(out / "events.geojson", dict(type="FeatureCollection", features=[dict(type="Feature",
        properties=e, geometry=dict(type="Point", coordinates=[e["longitude_deg"], e["latitude_deg"]]))
        for e in events if e["gps_valid"]]))


def validate_export(out, x, mask, source_time, patches, native, update_count):
    """Read artifacts back and assert coverage, units, clocks, and exact inputs."""
    frame = pd.read_parquet(out / "samples.parquet")
    n = len(x)
    assert len(frame) == n and frame.sample_index.tolist() == list(range(n))
    np.testing.assert_array_equal(frame.source_time_s, source_time)
    for j, channel in enumerate(CHANNELS):
        np.testing.assert_array_equal(frame[channel + "_valid"], mask[:, j])
        np.testing.assert_array_equal(frame[channel].fillna(0).to_numpy(dtype=np.float32), x[:, j])
    assert frame["time_s"].is_monotonic_increasing and frame["prediction_available_s"].is_monotonic_increasing
    assert (frame.prediction_available_s >= frame.time_s).all()
    assert frame.target_patch.nunique() == (n + PATCH - 1) // PATCH
    expected_final = max(0, n // PATCH - 2) * PATCH
    assert int(frame.is_final.sum()) == expected_final
    assert frame.loc[~frame.is_final, "disturbance"].isna().all()
    assert frame.loc[~frame.gps_valid, ["latitude_deg", "longitude_deg"]].isna().all().all()
    for row in patches:
        assert row["available_s"] >= available_time(row["emitted_after_samples"], native["input_ready_s"])
        if row["is_final"]:
            assert row["emitted_after_samples"] == row["target_sample_end"] + 2 * PATCH
    updates = 0
    last_available = -np.inf
    final_ids = []
    with gzip.open(out / "updates.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            assert row["available_s"] >= last_available
            last_available = row["available_s"]
            if row["is_final"]:
                final_ids.append(row["target_patch"])
            updates += 1
    assert final_ids == list(range(expected_final // PATCH))
    assert updates == update_count
    return dict(passed=True, samples=n, sample_coverage=1., final_samples=expected_final,
                provisional_samples=n - expected_final, valid_prediction_samples=int(frame.prediction_valid.sum()),
                gps_valid_samples=int(frame.gps_valid.sum()), gps_missing_or_stale_samples=int((~frame.gps_valid).sum()),
                max_input_interpolation_delay_s=float(np.max(native["input_ready_s"] - np.arange(n) / HZ)),
                replay_updates=updates, exact_model_input_roundtrip=True,
                final_predictions_emitted_once=True, no_future_gps=True)


def export_record(model, record, root, out, config):
    folder = root / record["path"]
    out.mkdir(parents=True, exist_ok=False)
    x, mask, source_time = [np.load(folder / f"{name}.npy") for name in ("x", "mask", "time")]
    assert len(x) == record["samples"] and np.isfinite(x).all()
    np.testing.assert_allclose(np.diff(source_time), 1 / HZ, atol=1e-9, rtol=0)
    native = native_sources(record, folder, source_time)
    start = time.perf_counter()
    update_count = 0
    with gzip.open(out / "updates.jsonl.gz", "wt", compresslevel=5) as handle:
        def save_update(row):
            nonlocal update_count
            value = enrich(row, record["id"], native["input_ready_s"], native["origin_ns"], float(source_time[0]), native["fixes"])
            handle.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
            update_count += 1
        raw_patches, finish = infer_drive(model, x, mask, config, save_update)
    inference_seconds = time.perf_counter() - start
    patches = [enrich(row, record["id"], native["input_ready_s"], native["origin_ns"], float(source_time[0]), native["fixes"]) for row in raw_patches]
    samples = sample_table(x, mask, source_time, native, patches)
    write_frame(out / "samples", samples)
    write_frame(out / "patches", pd.DataFrame(patches))
    write_native(out, native, float(source_time[0]))
    events = event_table(patches, finish)
    write_json(out / "events.json", events)
    geojson(out, native["fixes"], events)
    write_ground_truth(out, folder)
    audit = validate_export(out, x, mask, source_time, patches, native, update_count)
    summary = dict(session_id=record["id"], dataset=record["dataset"], split="test", samples=len(x),
                   duration_s=len(x) / HZ, patches=len(patches), events=len(events),
                   mapped_events=sum(e["gps_valid"] for e in events),
                   timestamp_origin_unix_ns=str(native["origin_ns"]),
                   timestamp_origin_utc=datetime.fromtimestamp(native["origin_ns"] / 1e9, timezone.utc).isoformat(),
                   source_time_start_s=float(source_time[0]), input_policy=native["input_policy"],
                   gps_source=native["gps_source"], gps_fixes=len(native["fixes"]),
                   unavailable_channels=[c for j, c in enumerate(CHANNELS) if not mask[:, j].any()],
                   source_record=record, finish=finish, audit=audit,
                   inference_and_update_serialization_seconds=inference_seconds)
    write_json(out / "session.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--ensemble", type=Path, default=RECEIPT)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This full ensemble export requires CUDA")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = read(args.data / "manifest.json")
    records = [r for r in manifest["records"] if r["split"] == "test" and r["source"] == "real" and r["dataset"] in ("kaggle", "lira_cd")]
    config = read(args.selection)["config"]
    if config.get("delay", 2) != 2:
        raise ValueError("Coverage audit expects the frozen two-patch delay")
    plan = dict(status="running", ensemble_path=str(args.ensemble.resolve()), ensemble_sha256=sha(args.ensemble),
                checkpoints=read(args.ensemble)["checkpoints"], manifest_path=str((args.data / "manifest.json").resolve()),
                manifest_sha256=sha(args.data / "manifest.json"), selection_path=str(args.selection.resolve()),
                selection_sha256=sha(args.selection), config=dict(delay=2, **config),
                postprocessing_policy="Frozen overlap consensus, then configured score filter on finalized patches only. No TEST tuning.",
                alert_filter=config.get("alert_filter"),
                sources_verified={}, source_code_sha256={str(p.relative_to(REPO)): sha(p) for p in
                    [Path(__file__), REPO / "road_training/timeline.py", REPO / "road_training/timeline_stream.py",
                     REPO / "road_training/alert_filter.py", REPO / "road_training/checkpoints.py"]},
                datasets=["kaggle", "lira_cd"], sessions=[r["id"] for r in records],
                samples=sum(r["samples"] for r in records), device=torch.cuda.get_device_name(),
                precision="CUDA bfloat16 autocast, float32 normalization; batch 1", selection="Every sample of every real Kaggle/LiRA TEST record")
    write_json(args.output / "plan.json", plan)
    for record in records:
        if record["source_file"] not in plan["sources_verified"]:
            digest = sha(record["source_file"])
            assert digest == record["source_sha256"]
            plan["sources_verified"][record["source_file"]] = digest
        for name, digest in record["file_sha256"].items():
            assert sha(args.data / record["path"] / name) == digest
    write_json(args.output / "plan.json", plan)
    model = load_teachers(args.ensemble, device="cuda")
    sessions = []
    for record in records:
        print(f"Exporting {record['id']}: {record['samples']:,} samples", flush=True)
        summary = export_record(model, record, args.data, args.output / record["id"], config)
        sessions.append(summary)
        write_json(args.output / "progress.json", dict(completed=[s["session_id"] for s in sessions], total=len(records)))
        print(json.dumps(dict(session=summary["session_id"], audit=summary["audit"], seconds=summary["inference_and_update_serialization_seconds"])), flush=True)
    result = dict(version=1, status="complete", sample_rate_hz=HZ, model_context_samples=1024,
                  patch_samples=PATCH, final_delay_after_patch_end_s=.32, gps_max_age_s=3.,
                  available_time_policy="Sensor data availability, includes native interpolation wait where needed; excludes compute, transport, map rendering. A sample interval ends at (index+1)/100.",
                  coordinate_system="Measured latitude/longitude degrees; GeoJSON order is longitude, latitude",
                  channels=CHANNELS, units=["m/s^2"] * 3 + ["rad/s"] * 3 + ["m/s"],
                  model_input_channels=channel_names(model_channels(model)),
                  alert_filter=config.get("alert_filter"),
                  score_filter_applied=config.get("alert_filter") is not None,
                  quality_bins_m_per_km=[2, 4, 6], quality_names=GRADES,
                  quality_bin_policy="Existing project display bins; not a validated universal road condition standard",
                  head_semantics=dict(disturbance="Binary localized disturbance, including manholes, depressions, bumps, cracks; not a dedicated pothole classifier", roughness="Estimated overall IRI, m/km"),
                  task_supervision=dict(kaggle="Disturbance labels; no measured IRI", lira_cd="Measured section IRI; no disturbance ground truth"),
                  normalization="Per-window masked instance normalization within the loaded models; exported sensors are unnormalized",
                  missing_policy="Null sensor values plus per-channel masks. Null GPS before first fix and after 3s staleness. Missing labels are not normal road.",
                  provisional_policy="Last two whole patches remain provisional. Incomplete final patch has an explicitly provisional masked EOF estimate; no hysteresis label, event ID, or fabricated future.",
                  samples=sum(s["samples"] for s in sessions), duration_s=sum(s["duration_s"] for s in sessions),
                  final_samples=sum(s["audit"]["final_samples"] for s in sessions),
                  provisional_samples=sum(s["audit"]["provisional_samples"] for s in sessions),
                  sessions=[{k: v for k, v in s.items() if k not in ("source_record", "finish")} for s in sessions])
    write_json(args.output / "manifest.json", result)
    artifacts = {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    write_json(args.output / "checksums.json", artifacts)
    print(json.dumps(dict(status="complete", samples=result["samples"], duration_s=result["duration_s"], output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
