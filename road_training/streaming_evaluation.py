"""Common patch, section and event evaluation for teachers and causal students."""
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import average_precision_score
from road_training.dataset import RoadDataset
from road_training.common import ROOT
from road_training.checkpoints import load_teachers, load_student

DATA = ROOT / "road_training/data_with_roadsens"
from road_training.train_multitask import supervised_windows, patch_targets, EpochScores, joint_loss
from road_training.streaming_data import ContextWindows
from road_training.streaming_model import Ensemble, StreamingRoadModel, aligned_output
from road_training.common import read, sha
from road_training.metrics import section_metrics, scored_events, intervals, match_events


def validation_loader(split, batch_size=256):
    data = RoadDataset(DATA, split=split, source='real', stride=1024, return_labels=True)
    selected, _, _ = supervised_windows(data, 16)
    wrapped = ContextWindows(data)
    loader = DataLoader(Subset(wrapped, selected.indices), batch_size=batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)
    return data, loader


def predict(model, batch, device='cuda', live_teacher=False):
    """Return predictions aligned to the unchanged 64 target patches.

    All methods share the requirement of 2 actually observed following patches,
    censoring only unavailable terminal emissions, not annotation boundaries.
    live_teacher runs a 64-patch rolling window, emitting target at index 61.
    """
    x, mask = batch['context_x'].to(device), batch['context_mask'].to(device)
    if isinstance(model, StreamingRoadModel):
        output = aligned_output(model(x, mask), 48, model.delay_patches)
    elif live_teacher:
        raise NotImplementedError('Rolling teacher uses the recording evaluator')
    else:
        output = model(batch['x'].to(device), batch['mask'].to(device))
    output['patch_valid'] = output['patch_valid'] & batch['context_available'][:, 50:114].to(device)
    return output


@torch.no_grad()
def evaluate(model, split='val', loader=None, data=None, detailed=True):
    model.eval()
    if loader is None:
        data, loader = validation_loader(split)
    device = next(model.parameters()).device
    scores = EpochScores(device)
    roughness, disturbance = defaultdict(list), defaultdict(list)
    for batch in loader:
        targets = {k:v.to(device) for k,v in patch_targets(batch, 16).items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
            output = predict(model, batch, device)
        losses = joint_loss(output, targets)
        scores.add(output, targets, losses)
        if not detailed:
            continue
        pred = output['roughness'].cpu().numpy()
        probability = output['disturbance_logit'].float().sigmoid().cpu().numpy()
        starts = batch['time'].numpy()[:, ::16]
        section = batch['labels']['roughness_section'].numpy()[:, ::16]
        for i, recording in enumerate(batch['recording_id']):
            rv, dv = losses['roughness_valid'][i].cpu().numpy(), losses['disturbance_valid'][i].cpu().numpy()
            for j in np.flatnonzero(rv):
                roughness[recording, int(section[i,j])].append((float(pred[i,j]), float(targets['roughness'][i,j])))
            if batch['dataset'][i] == 'kaggle':
                speed = batch['x'][i,:,6].numpy().reshape(-1,16)
                speed_valid = batch['mask'][i,:,6].numpy().reshape(-1,16).all(1)
                truth = targets['disturbance'][i].cpu().numpy()
                for j in range(64):
                    disturbance[recording].append((starts[i,j], probability[i,j], dv[j],
                        float(speed[j].mean())*.16 if speed_valid[j] and dv[j] else 0., int(truth[j])))
    result = scores.result(1., 1.)
    result.update(split=split, threshold=.5, coverage='Common target patches with two subsequent complete recorded patches available')
    if not detailed:
        return result
    lookup = {r['id']:r for r in data.records}
    rows = []
    for (recording, section), values in roughness.items():
        a = np.array(values)
        rows.append(dict(recording=recording, road=lookup[recording]['groups'][0], section=section,
                         prediction=float(a[:,0].mean()), target=float(a[:,1].mean()), patches=len(a)))
    result['sections'] = section_metrics(rows)
    events = []
    for recording, values in disturbance.items():
        a = np.array(values); a = a[np.argsort(a[:,0])]
        annotations = read(DATA/lookup[recording]['path']/'metadata.json')['annotations']
        scored = scored_events(a[:,0], a[:,2].astype(bool), annotations)
        predictions = intervals(a[:,0], a[:,1]>=.5, scored['valid'])
        matching = match_events(predictions, scored['reference'])
        km = float(a[scored['valid'],3].sum()/1000)
        valid = a[:,2].astype(bool)
        delay = model.delay_patches*.16 if isinstance(model, StreamingRoadModel) else None
        detection_delays = []
        if delay is not None:
            # Actual alert timestamp, without subtracting the model's delay.
            for i,j in matching['pairs']:
                detection_delays.append(predictions[i][0]+.16+delay-scored['reference'][j][0])
        events.append(dict(recording=recording, **matching, false_alerts_per_km=matching['fp']/km if km else None,
            scored_distance_km=km, average_precision=float(average_precision_score(a[valid,4], a[valid,1])),
            predicted_intervals=predictions, reference_intervals=scored['reference'],
            censored=scored['censored'], matched_alert_minus_reference_onset_seconds=detection_delays))
    result['events'] = events
    return result


def f1(metrics):
    return metrics['disturbance']['classification']['per_class']['disturbance']['f1']
