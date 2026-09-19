"""Choose alert filtering on VAL, then evaluate frozen candidates on TEST."""
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from road_training.alert_filter import AlertPostprocessor
from road_training.export_test_drives import DATA, RECEIPT, SELECTION, OUTPUT as EXPORT, REPO, infer_drive, native_sources, available_time, read, sha, write_json
from road_training.streaming_evaluation import load_teachers
from road_training.metrics import intervals, scored_events, match_events


OUT = REPO / "reports/alert_filter_20260919"
BASE = dict(kind="none", space="probability", onset=.6, offset=.5)


def configs():
    filters = [dict(kind="none", space="probability")]
    for space in ("probability", "logit"):
        for alpha in (.8, .6, .4):
            filters.append(dict(kind="ema", space=space, alpha=alpha))
            # Match asymptotic gains so differences aren't just unfair tuning.
            filters.append(dict(kind="kalman", space=space, q_over_r=alpha**2/(1-alpha)))
    result = []
    for f in filters:
        for onset in (.6, .65, .7, .75, .8, .85):
            for gap in (.1, .2):
                result.append(dict(**f, onset=onset, offset=round(onset-gap, 3)))
    return result


def process(rows, config):
    state = AlertPostprocessor(**config)
    return [state.update(row) for row in rows]


def truth(record):
    folder = DATA / record["path"]
    labels = np.load(folder / "labels.npy")[:, 1]
    observed = np.load(folder / "mask.npy").any(1)
    whole = len(labels) // 16
    y = labels[:whole*16].reshape(-1, 16)
    known = (labels[:whole*16] >= 0).reshape(-1, 16) & observed[:whole*16].reshape(-1, 16)
    ones, zeros = [((y == c) & known).sum(1) for c in (1, 0)]
    x = np.load(folder / "x.npy")
    return dict(y=ones > zeros, valid=(ones != zeros),
                annotations=read(folder / "metadata.json")["annotations"],
                origin=float(np.load(folder / "time.npy")[0]),
                meters=x[:whole*16, 6].reshape(-1, 16).mean(1)*.16)


def evaluate(rows, gt):
    # Score every final patch, not only the previously selected full windows.
    final = [row for row in rows if row["is_final"]]
    index = np.array([row["target_patch"] for row in final])
    valid = np.array([row["valid"] for row in final], bool)
    predicted = np.array([bool(row["disturbance"]) for row in final])
    starts = np.array([row["start_s"] for row in final]) + gt["origin"]
    use = gt["valid"][index] & valid
    y = gt["y"][index]
    tp = int((use & y & predicted).sum()); fp = int((use & ~y & predicted).sum()); fn = int((use & y & ~predicted).sum())
    scored = scored_events(starts, valid, gt["annotations"])
    segments = intervals(starts, predicted, scored["valid"])
    matches = match_events(segments, scored["reference"])
    # Event onset is a target time; alert time is the actual exported availability.
    lookup = {round(row["start_s"] + gt["origin"], 6): row["available_s"] + gt["origin"] for row in final}
    delays = [lookup[round(segments[i][0], 6)] - scored["reference"][j][0] for i, j in matches["pairs"]]
    by_type = Counter()
    detected_types = Counter()
    matched_gt = {j for _, j in matches["pairs"]}
    for j, kinds in enumerate(scored["types"]):
        for kind in set(kinds):
            by_type[kind] += 1
            detected_types[kind] += int(j in matched_gt)
    km = float(gt["meters"][index][scored["valid"]].sum()/1000)
    return dict(patch_precision=tp/max(tp+fp, 1), patch_recall=tp/max(tp+fn, 1),
                patch_f1=2*tp/max(2*tp+fp+fn, 1), patch_tp=tp, patch_fp=fp, patch_fn=fn,
                events={k:v for k,v in matches.items() if k != "pairs"},
                predicted_events=len(segments), reference_events=len(scored["reference"]),
                false_alerts_per_km=matches["fp"]/km if km else None,
                distance_km=km, disturbance_fraction=float(predicted[valid].mean()),
                onset_delay_median_s=float(np.median(delays)) if delays else None,
                onset_delay_p95_s=float(np.percentile(delays,95)) if delays else None,
                recall_by_type={k:dict(detected=detected_types[k], reference=n, recall=detected_types[k]/n) for k,n in by_type.items()})


def acceptable(candidate, baseline):
    return (candidate["events"]["recall"] >= baseline["events"]["recall"] - .05
            and candidate["patch_recall"] >= baseline["patch_recall"] - .05
            and candidate["events"]["f1"] >= baseline["events"]["f1"] - .01
            and candidate["patch_f1"] >= baseline["patch_f1"] - .02)


def rank(row):
    m = row["metrics"]
    return (m["events"]["precision"], m["events"]["f1"], m["patch_f1"], -m["events"]["fp"])


def collect_validation(record):
    folder = DATA / record["path"]
    x, mask, times = [np.load(folder / f"{name}.npy") for name in ("x", "mask", "time")]
    native = native_sources(record, folder, times)
    model = load_teachers(RECEIPT)
    rows, _ = infer_drive(model, x, mask, read(SELECTION)["config"])
    for row in rows:
        row.update(is_final=row["status"] == "final", start_s=row["target_sample_start"]/100,
                   end_s=row["target_sample_end"]/100,
                   available_s=available_time(row["emitted_after_samples"], native["input_ready_s"]))
        if not row["is_final"]:
            row.update(disturbance=None, event_transition=None, event_id=None)
    write_json(OUT / "val_predictions.json", rows)
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4); torch.set_float32_matmul_precision("high"); torch.manual_seed(0)
    source = [Path(__file__), Path(__file__).resolve().parents[2] / 'road_training/alert_filter.py']
    plan = dict(ensemble_sha256=sha(RECEIPT), checkpoints=read(RECEIPT)["checkpoints"],
                manifest_sha256=sha(DATA / "manifest.json"), source_sha256={str(p):sha(p) for p in source},
                candidates=configs(), baseline=BASE, selection_split="val",
                selection="Maximize event precision subject to event/patch recall losses <=5 percentage points, event F1 loss <=.01 and patch F1 loss <=.02 versus baseline; tie-break event F1, patch F1, fewer false alerts.",
                temporal_policy="Forward-only scalar filter of once-finalized consensus scores. Reset at missing input and session boundaries. No extra future buffer, backdating, minimum duration, or IRI changes.",
                test_policy="Previously viewed test data; frozen VAL decisions evaluated once. This is not a newly untouched holdout.",
                literature=["https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf", "https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm"])
    write_json(OUT / "plan.json", plan)
    manifest = read(DATA / "manifest.json")
    records = {r["split"]:r for r in manifest["records"] if r["dataset"] == "kaggle" and r["split"] in ("val","test")}
    print("Collecting full VAL drive with current four-model ensemble", flush=True)
    val_rows = collect_validation(records["val"]); val_gt = truth(records["val"])
    baseline = evaluate(process(val_rows, BASE), val_gt)
    candidates = []
    for config in plan["candidates"]:
        result = evaluate(process(val_rows, config), val_gt)
        candidates.append(dict(config=config, metrics=result, acceptable=acceptable(result, baseline)))
    write_json(OUT / "validation_search.json", dict(baseline=baseline, candidates=candidates))
    champions = {}
    for kind in ("none", "ema", "kalman"):
        family = [r for r in candidates if r["config"]["kind"] == kind]
        eligible = [r for r in family if r["acceptable"]]
        champions[kind] = max(eligible, key=rank) if eligible else max(family, key=lambda r:r["metrics"]["events"]["f1"])
    selected = max([r for r in candidates if r["acceptable"]], key=rank)
    decision = dict(selected=selected, family_champions=champions, baseline=baseline,
                    test_read_this_run=False, plan_sha256=sha(OUT / "plan.json"),
                    validation_predictions_sha256=sha(OUT / "val_predictions.json"))
    write_json(OUT / "selection.json", decision)
    print("Frozen VAL selection: " + json.dumps(selected), flush=True)

    # No TEST scores have entered the search or selection above.
    folder = EXPORT / records["test"]["id"]
    test_rows = pd.read_parquet(folder / "patches.parquet").to_dict("records")
    test_gt = truth(records["test"])
    profiles = dict(baseline=BASE, thresholds=champions["none"]["config"], ema=champions["ema"]["config"],
                    kalman=champions["kalman"]["config"], selected=selected["config"])
    results = {name:dict(config=config, validation=evaluate(process(val_rows, config), val_gt),
                        test=evaluate(process(test_rows, config), test_gt)) for name,config in profiles.items()}
    write_json(OUT / "results.json", results)
    # Save selected display updates separately, preserving immutable exports.
    generated = []
    for session in read(EXPORT / "manifest.json")["sessions"]:
        processor = AlertPostprocessor(**selected["config"])
        target = OUT / "selected_updates" / (session["session_id"] + ".jsonl.gz")
        target.parent.mkdir(exist_ok=True)
        count = 0
        with gzip.open(EXPORT / session["session_id"] / "updates.jsonl.gz", "rt") as src, gzip.open(target, "wt") as dst:
            for line in src:
                original = json.loads(line); row = processor.update(original)
                assert row["available_s"] == original["available_s"]
                assert row["iri_m_per_km"] == original["iri_m_per_km"]
                dst.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                count += 1
        generated.append(dict(session=session["session_id"], updates=count, sha256=sha(target)))
    write_json(OUT / "complete.json", dict(completed=True, test_evaluated=True, selected=selected["config"],
               selection_sha256=sha(OUT / "selection.json"), results_sha256=sha(OUT / "results.json"), derived_updates=generated))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
