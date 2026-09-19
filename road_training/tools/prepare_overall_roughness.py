"""Add preserved full-road section IRI targets to the prepared synthetic data.

Reads saved geometry-derived targets, not sensor-derived pseudo-labels. Requires
the original section CSV hash and unique sample-to-section alignment. No physics
or road reconstruction is run. Existing real targets are untouched. Runtime Dataset code
needs only the exported NPY arrays, never the simulator.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def align_sections(background, valid, sections):
    """Map source-aligned section membership to total IRI, rejecting ambiguity."""
    iri = np.full(len(background), np.nan, np.float32)
    ids = np.full(len(background), -1, np.int32)
    for i, section in enumerate(sections):
        value = float(section["realized_iri_m_per_km"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("Section IRI must be finite and nonnegative")
        take = valid & np.isclose(background, float(section["background_iri_m_per_km"]),
                                  rtol=1e-6, atol=1e-6)
        if (ids[take] >= 0).any():
            raise ValueError("Background IRI does not uniquely identify the section")
        iri[take], ids[take] = value, i
    if (valid & (ids < 0)).any():
        raise ValueError("Some valid samples have no matching section")
    return iri, ids


def prepare(root):
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prepared = []
    for record in manifest["records"]:
        if record["source"] != "synthetic":
            continue
        folder = root / record["path"]
        meta = json.loads((folder / "metadata.json").read_text())
        csv_path = Path(meta["raw_directory"]) / "road_segments.csv"
        expected = meta["metadata"]["files_sha256"]["road_segments.csv"]
        if checksum(csv_path) != expected:
            raise ValueError(f"Original section table changed: {csv_path}")
        with csv_path.open(newline="") as handle:
            sections = list(csv.DictReader(handle))
        with np.load(folder / "original_targets.npz", allow_pickle=False) as original:
            iri, ids = align_sections(original["background_iri"], original["background_valid"], sections)
            if not np.array_equal(original["time_s"], np.load(folder / "time.npy")):
                raise ValueError(f"Target and input clocks disagree: {record['id']}")
        if len(iri) != record["samples"]:
            raise ValueError(f"Target and input lengths disagree: {record['id']}")
        # Validate the entire corpus before writing any new targets.
        prepared.append((record, folder, iri, ids, dict(
            source_csv=str(csv_path), source_csv_sha256=expected, sections=sections,
            definition="Realized section IRI in m/km including injected geometry on nominal +/-0.8 m wheel tracks; not exact driven wheel paths.",
            alignment="Original source-time background validity and unique section IRI membership; transitions remain unknown.")))
    for record, folder, iri, ids, provenance in prepared:
        for name, array in [("overall_iri", iri), ("roughness_section", ids)]:
            path = folder / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            record["file_sha256"][path.name] = checksum(path)
        path = folder / "overall_roughness.json"
        path.write_text(json.dumps(provenance, indent=2) + "\n")
        record["file_sha256"][path.name] = checksum(path)
    manifest["overall_roughness"] = dict(
        units="m/km", target="Realized section IRI including injected geometry",
        source="Original hash-verified road_segments.csv; source-time section membership",
        real_labels=("LiRA: measured P79 section IRI. Kaggle: unknown." if any(
            r.get("dataset")=="lira_cd" for r in manifest["records"]) else
            "Unavailable in this package; masked, never inferred from acceleration or defect types"),
        prediction="One estimate per patch for its containing road section; not IRI measured independently over a 0.16-second patch",
        synthetic_records=len(prepared))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summary = dict(records=len(prepared), valid_samples=sum(int(np.isfinite(x[2]).sum()) for x in prepared),
                   all_original_section_hashes_verified=True)
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    prepare(parser.parse_args().data_root)
