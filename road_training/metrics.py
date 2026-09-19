"""Section and event metrics complementing the existing patch scores."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from road_training.dataset import RoadDataset
from road_training.patchtst import PatchTST, PatchTSTRoadModel
from road_training.train_multitask import patch_targets, supervised_windows
from road_training.common import read, sha, write


def intervals(starts, positive, valid, patch_seconds=.16):
    """Merge touching positive patches; never bridge unknown gaps."""
    starts, positive, valid = np.asarray(starts), np.asarray(positive), np.asarray(valid)
    order = np.argsort(starts)
    result = []
    for t, keep in zip(starts[order], (positive & valid)[order]):
        if not keep:
            continue
        if result and abs(result[-1][1]-t) < 1e-6:
            result[-1][1] = float(t+patch_seconds)
        else:
            result.append([float(t), float(t+patch_seconds)])
    return result


def match_events(predicted, reference, tolerance_s=.25):
    """Maximum-cardinality one-to-one matching of overlapping/tolerant intervals."""
    predicted = np.asarray(predicted, float).reshape(-1, 2)
    reference = np.asarray(reference, float).reshape(-1, 2)
    if tolerance_s < 0 or not np.isfinite(tolerance_s):
        raise ValueError("Finite nonnegative event tolerance required")
    for array in (predicted, reference):
        if not np.isfinite(array).all() or np.any(array[:, 1] <= array[:, 0]):
            raise ValueError("Positive finite event intervals required")
    pairs = []
    if len(predicted) and len(reference):
        overlap = np.minimum(predicted[:, None, 1], reference[None, :, 1]) - np.maximum(predicted[:, None, 0], reference[None, :, 0])
        allowed = overlap >= -tolerance_s
        # Each extra valid match dominates every possible tie-break gain.
        quality = np.maximum(overlap, 0)/(1+np.maximum(overlap, 0))
        reward = allowed.astype(float)*(min(len(predicted), len(reference))+1) + quality*allowed
        rows, cols = linear_sum_assignment(-reward)
        pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if allowed[i, j]]
    tp = len(pairs);fp = len(predicted)-tp;fn = len(reference)-tp
    precision = tp/max(tp+fp, 1);recall = tp/max(tp+fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2*tp/max(2*tp+fp+fn, 1), pairs=pairs, tolerance_s=tolerance_s)


def scored_events(starts, valid, annotations, patch_seconds=.16, tolerance_s=.25):
    """Censor incomplete annotations and their prediction region symmetrically.

    A GT event must be fully covered by one contiguous scored interval. Events
    crossing missing data or a dropped final input window cannot become false
    negatives, and predictions at their censored portion cannot become false
    positives. Overlapping type annotations form one binary disturbance event.
    """
    starts=np.asarray(starts,float);keep=np.asarray(valid,bool).copy()
    if starts.shape!=keep.shape or np.any(np.diff(starts)<=0):
        raise ValueError('Sorted unique patch starts and aligned validity required')
    events=[]
    for annotation in sorted(annotations,key=lambda a:a['rel_t_start']):
        if annotation['anomaly'] not in ('manhole','depression','bump','crack'):continue
        lo,hi=annotation['rel_t_start'],annotation['rel_t_end']
        if not np.isfinite([lo,hi]).all() or hi<=lo:raise ValueError('Invalid annotation interval')
        if events and lo<=events[-1][1]:
            events[-1][1]=max(hi,events[-1][1]);events[-1][2].append(annotation['anomaly'])
        else:events.append([lo,hi,[annotation['anomaly']]])
    excluded=set()
    while True:
        coverage=intervals(starts,np.ones(len(starts),bool),keep,patch_seconds)
        newly={i for i,(lo,hi,_) in enumerate(events) if not any(lo>=a-1e-7 and hi<=b+1e-7 for a,b in coverage)}-excluded
        if not newly:break
        for i in newly:
            lo,hi,_=events[i]
            keep &= ~((starts<hi+tolerance_s)&(starts+patch_seconds>lo-tolerance_s))
        excluded|=newly
    included=[event for i,event in enumerate(events) if i not in excluded]
    return dict(valid=keep,reference=[e[:2] for e in included],types=[e[2] for e in included],
                censored=[e for i,e in enumerate(events) if i in excluded])


def section_metrics(rows):
    """Equal-weight traversals, then equal-weight spatial sections."""
    sections = defaultdict(list)
    for row in rows:
        sections[(row["road"], row["section"])].append(row)
    output = []
    for (road, section), group in sorted(sections.items()):
        target = np.array([r["target"] for r in group])
        if np.ptp(target) > 1e-4:
            raise ValueError("Repeated section has inconsistent reference targets")
        prediction = np.array([r["prediction"] for r in group])
        output.append(dict(road=road, section=section, target=float(target.mean()),
            prediction=float(prediction.mean()), traversal_std=float(prediction.std()), traversals=len(group)))
    error = np.array([r["prediction"]-r["target"] for r in output])
    ordinal=None
    if len(output):
        target=np.array([r['target'] for r in output]);prediction=np.array([r['prediction'] for r in output])
        truth_bin=np.digitize(target,[2.,4.,6.]);pred_bin=np.digitize(prediction,[2.,4.,6.])
        matrix=np.zeros((4,4),int);np.add.at(matrix,(truth_bin,pred_bin),1)
        support=matrix.sum(1);predicted=matrix.sum(0);tp=np.diag(matrix)
        precision=np.divide(tp,predicted,out=np.zeros(4,float),where=predicted>0)
        recall=np.divide(tp,support,out=np.zeros(4,float),where=support>0)
        f1=np.divide(2*tp,support+predicted,out=np.zeros(4,float),where=support+predicted>0)
        present=support>0
        ordinal=dict(thresholds_m_per_km=[2.,4.,6.],names=['good','medium','bad','terrible'],
            definition='Existing project IRI bins, not a separate human road-quality annotation or official ISO class',
            confusion=matrix.tolist(),accuracy=float(np.mean(truth_bin==pred_bin)),
            mean_absolute_class_error=float(np.mean(abs(truth_bin-pred_bin))),
            reference_support=support.tolist(),
            per_class=[dict(precision=float(precision[i]),recall=float(recall[i]) if present[i] else None,
                f1=float(f1[i]) if present[i] else None,support=int(support[i])) for i in range(4)],
            macro_f1_present_classes=float(f1[present].mean()),
            balanced_accuracy_present_classes=float(recall[present].mean()),
            constant_good_accuracy=float(support[0]/support.sum()),
            absent_class_policy='Absent reference classes have unknown recall/F1 and are excluded from macro averages; predictions into them still count as errors.')
    return dict(sections=len(output), mae=float(np.abs(error).mean()) if len(error) else None,
                rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                ordinal=ordinal,
                aggregation="Patches averaged within traversal, then traversals within section; equal spatial-section weights",
                rows=output)


def evaluate(checkpoint, output, split='val', *, encoder_class=PatchTST, model_class=PatchTSTRoadModel):
    if split not in ('val','test'):raise ValueError('Held-out evaluation requires val or test')
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"];root = Path(config["data_root"])
    if sha(root/"manifest.json") != config["manifest_sha256"]:
        raise ValueError("Checkpoint dataset changed")
    data = RoadDataset(root, source="real", split=split, stride=1024, window_size=1024, return_labels=True)
    selected, _, _ = supervised_windows(data, 16)
    model = model_class(encoder_class(**config["encoder_config"]))
    model.load_state_dict(saved["model_state"]);model.to("cuda").eval()
    loader = DataLoader(selected, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    roughness = defaultdict(list);disturbance = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            targets = patch_targets(batch, 16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(batch["x"].to("cuda"), batch["mask"].to("cuda"))
            pred = result["roughness"].cpu().numpy()
            probability = result["disturbance_logit"].float().sigmoid().cpu().numpy()
            starts = batch["time"].numpy()[:, ::16]
            section = batch["labels"]["roughness_section"].numpy()[:, ::16]
            for i, recording in enumerate(batch["recording_id"]):
                rv = targets["roughness_valid"][i].numpy()
                for j in np.flatnonzero(rv):
                    roughness[(recording, int(section[i, j]))].append((float(pred[i, j]), float(targets["roughness"][i, j])))
                if batch["dataset"][i] == "kaggle":
                    speed = batch["x"][i, :, 6].numpy().reshape(-1, 16)
                    speed_valid = batch["mask"][i, :, 6].numpy().reshape(-1, 16).all(1)
                    dv = targets["disturbance_valid"][i].numpy()
                    truth = targets["disturbance"][i].numpy()
                    for j in range(len(starts[i])):
                        disturbance[recording].append((starts[i, j], probability[i, j], dv[j],
                            float(speed[j].mean())*.16 if speed_valid[j] and dv[j] else 0.,int(truth[j])))
    record_lookup = {r["id"]:r for r in data.records}
    rows = []
    for (recording, section), values in roughness.items():
        values = np.array(values)
        rows.append(dict(recording=recording, road=record_lookup[recording]["groups"][0], section=section,
                         prediction=float(values[:, 0].mean()), target=float(values[:, 1].mean()), patches=len(values)))
    event_rows = []
    for recording, values in disturbance.items():
        a = np.array(values);order = np.argsort(a[:, 0]);a = a[order]
        annotations = read(root/record_lookup[recording]["path"]/"metadata.json")["annotations"]
        scored=scored_events(a[:,0],a[:,2].astype(bool),annotations)
        reference,types=scored['reference'],scored['types']
        prediction = intervals(a[:, 0], a[:, 1] >= .5, scored['valid'])
        matched = match_events(prediction, reference)
        patch_valid=a[:,2].astype(bool)
        patch_truth=a[patch_valid,4].astype(int)
        ranking=dict(average_precision=float(average_precision_score(patch_truth,a[patch_valid,1])) if patch_truth.any() else None,
            prevalence=float(patch_truth.mean()),labeled_patches=int(patch_valid.sum()),
            definition='Positive-class patch average precision over all valid patch targets; ranking metric, no threshold tuning')
        km = a[scored['valid'], 3].sum()/1000
        detected = {j for _, j in matched["pairs"]}
        by_type = {name:dict(events=sum(name in t for t in types), detected=sum(i in detected for i,t in enumerate(types) if name in t))
                   for name in sorted({t for group in types for t in group})}
        event_rows.append(dict(recording=recording, **matched, scored_distance_km=float(km),
                               false_alerts_per_km=matched["fp"]/km if km else None,
                               per_annotation_type=by_type, predicted_intervals=prediction, reference_intervals=reference,
                               patch_ranking=ranking,
                               censored_events=scored['censored'],reference_types=types))
    result = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), epoch=saved["epoch"],
        split=split, metric_version=2, threshold=.5, roughness=section_metrics(rows), roughness_traversals=rows,
        disturbance=event_rows, limitations=["One validation road per dataset; seed variability is not cross-road generalization.",
            "Event matching is one-to-one with fixed 0.25-s interval tolerance; patch metrics remain separately reported.",
            "Kaggle depression labels do not independently establish pothole-specific accuracy."])
    write(output, result)
    return result
