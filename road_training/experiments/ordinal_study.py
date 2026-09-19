"""Matched regression/ordinal x no-PVS/PVS fine-tuning, with frozen TEST access.

All arms start from the same trained four-input members, retain mounting/noise
augmentation and the disturbance task, and see identical existing-data batches.
PVS adds weak ordinal supervision with separate vehicle-specific cutpoints.
It never supplies physical IRI targets or detector negatives.
"""
import argparse
from collections import defaultdict
import gc
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from road_training.acceleration_speed import AccelerationSpeedDataset, augment
from road_training.common import read, write, sha
from road_training.experiments.real_mixup import dataset_pools, sample
from road_training.ordinal import CUTPOINTS, NAMES, OrdinalRoadModel, iri_classes, ordinal_loss
from road_training.ordinal_data import PVSDataset, pvs_patch_targets
from road_training.train import balanced_alpha, focal_from_cross_entropy
from road_training.train_multitask import patch_targets, supervised_windows


ARMS = ('regression', 'ordinal', 'regression_pvs', 'ordinal_pvs')


def class_metrics(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=int), np.asarray(prediction, dtype=int)
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    support, predicted, tp = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    present = support > 0
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=present)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros(3), where=support+predicted > 0)
    return dict(confusion=matrix.tolist(), support=support.tolist(),
        accuracy=float((truth == prediction).mean()),
        macro_f1_present=float(f1[present].mean()), macro_f1_all_three=float(f1.mean()),
        balanced_accuracy_present=float(recall[present].mean()),
        mean_absolute_class_error=float(np.abs(truth-prediction).mean()),
        severe_error_fraction=float((np.abs(truth-prediction) == 2).mean()),
        per_class={name: dict(precision=float(precision[i]), recall=float(recall[i]) if present[i] else None,
                              f1=float(f1[i]) if present[i] else None, support=int(support[i]))
                   for i, name in enumerate(NAMES)})


def section_results(rows, mode):
    """Average patches within traversal, then equally weight traversals/sections."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['road'], row['section']].append(row)
    sections = []
    for (road, section), values in sorted(groups.items()):
        targets = np.array([r['target'] for r in values])
        if np.ptp(targets) > 1e-4:
            raise ValueError('Repeated physical section has inconsistent IRI')
        target = float(targets.mean())
        score = float(np.mean([r['score'] for r in values]))
        probability = np.mean([r['probability'] for r in values], axis=0)
        if mode == 'regression':
            prediction = int(iri_classes(torch.tensor(score, dtype=torch.float64)))
        else:
            # Median of the ordered distribution, consistent with cumulative decisions.
            prediction = int(probability[1:].sum() >= .5)+int(probability[2] > .5)
        sections.append(dict(road=road, section=section, target=target, score=score,
            truth=int(iri_classes(torch.tensor(target, dtype=torch.float64))), prediction=prediction,
            probability=probability.tolist(), traversals=len(values)))
    result = class_metrics([r['truth'] for r in sections], [r['prediction'] for r in sections])
    result.update(sections=len(sections), rows=sections)
    if mode == 'regression':
        result['iri_mae'] = float(np.mean([abs(r['score']-r['target']) for r in sections]))
    return result


def data_loader(root, split, *, batch_size=128):
    data = AccelerationSpeedDataset(root, source='real', split=split, stride=1024, return_labels=True,
                                    max_cached_recordings=128)
    selected, _, _ = supervised_windows(data, 16)
    return data, DataLoader(selected, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.inference_mode()
def evaluate(model, data, loader):
    model.eval()
    lookup = {r['id']: r for r in data.records}
    traversals = defaultdict(list)
    patch_truth, patch_pred = [], []
    detection = np.zeros((2, 2), dtype=np.int64)
    for batch in loader:
        target = patch_targets(batch, 16)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(batch['x'].cuda(), batch['mask'].cuda())
        valid = output['patch_valid'].cpu()
        score = output['quality_score'].float().cpu()
        probability = output['quality_probability'].float().cpu().numpy()
        classes = output['quality_class'].cpu()
        detector = output['disturbance_logit'].float().cpu().sigmoid()
        for i, recording in enumerate(batch['recording_id']):
            rv = target['roughness_valid'][i] & valid[i]
            patch_truth.extend(iri_classes(target['roughness'][i, rv]).tolist())
            patch_pred.extend(classes[i, rv].tolist())
            for j in rv.nonzero().flatten().tolist():
                section = int(batch['labels']['roughness_section'][i, j*16])
                traversals[recording, section].append((float(score[i, j]),
                    float(target['roughness'][i, j]), probability[i, j]))
            if batch['dataset'][i] == 'kaggle':
                dv = target['disturbance_valid'][i] & valid[i]
                y = target['disturbance'][i, dv].numpy()
                pred = (detector[i, dv].numpy() >= .5).astype(int)
                np.add.at(detection, (y, pred), 1)
    rows = []
    for (recording, section), values in traversals.items():
        rows.append(dict(road=lookup[recording]['groups'][0], section=section,
            score=float(np.mean([v[0] for v in values])), target=float(np.mean([v[1] for v in values])),
            probability=np.mean([v[2] for v in values], axis=0).tolist()))
    tp, fp, fn = int(detection[1, 1]), int(detection[0, 1]), int(detection[1, 0])
    return dict(sections=section_results(rows, model.mode), patches=class_metrics(patch_truth, patch_pred),
                disturbance=dict(confusion=detection.tolist(), precision=tp/max(1, tp+fp),
                    recall=tp/max(1, tp+fn), f1=2*tp/max(1, 2*tp+fp+fn)))


def initialize(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise FileExistsError('Preserve the existing ordinal experiment plan')
    receipt = read(args.ensemble)
    files = list(receipt['checkpoints'])
    checkpoints = {}
    for seed in args.seeds:
        index = receipt['seeds'].index(seed)
        path = (args.ensemble.parent/files[index]).resolve()
        if sha(path) != receipt['checkpoints'][files[index]]:
            raise ValueError('Source checkpoint changed')
        checkpoints[str(seed)] = dict(path=str(path), sha256=sha(path))
    root = Path(__file__).resolve().parents[2]
    source_files = [Path(__file__), *[root/'road_training'/name for name in (
        'ordinal.py', 'ordinal_data.py', 'acceleration_speed.py', 'instance_model.py', 'patchtst.py',
        'mounting_augmentation.py', 'augmentation.py', 'dataset.py', 'train_multitask.py',
        'train.py', 'experiments/real_mixup.py', 'tools/prepare_pvs.py')]]
    plan = dict(arms=ARMS, seeds=args.seeds, initial_checkpoints=checkpoints,
        freeze_backbone=args.freeze_backbone,
        data=str(args.data.resolve()), pvs=str(args.pvs.resolve()),
        manifest_sha256=sha(args.data/'manifest.json'), pvs_manifest_sha256=sha(args.pvs/'manifest.json'),
        cutpoints_m_per_km=CUTPOINTS, names=NAMES,
        thresholds='Good <95 in/mi, medium 95..170, bad >170; exact 63.36 conversion, unrounded IRI. FHWA-inspired IRI-only research categories, not whole pavement condition.',
        threshold_source='https://www.fhwa.dot.gov/tpm/guidance/hif18022.pdf',
        ordinal_method='Two ordered cumulative BCE tasks on log positive score, fixed physical cutpoints and learned temperature. Inspired by rank-consistent ordinal regression; not an exact CORAL benchmark reproduction.',
        ordinal_source='https://arxiv.org/abs/1901.07884',
        epochs=8, steps_per_epoch=96, batch_size=128, pvs_batch_size=32,
        samples_per_batch=dict(kaggle=32, lira=64, roadsens=32),
        backbone_lr=5e-6, head_lr=1e-4, weight_decay=.01, quality_weight=.25, pvs_weight=.10,
        precision='cuda bf16 with FP32 instance normalization and losses',
        augmentation='TRAIN mounting yaw +/-20 degrees, tilt +/-10 degrees, probability .75; existing held-sample noise/bias; speed unchanged',
        initialization='Same already-trained four-input regression seed for each paired arm; all existing tensors match. No changes to deployed checkpoints.',
        sampling='Identical eligible stride-one existing-data windows and augmentation draws per seed/epoch; PVS draws use a separate RNG. Equal optimizer-update budget; PVS arms add 32 windows per update.',
        quality_weighting='Same inverse physical TRAIN-section observation count for regression and ordinal; exclude section-boundary patches',
        selection='Maximum LiRA VAL spatial-section macro F1 among present classes, then minimum class MAE; require Kaggle VAL detector F1 within .02 of that seed baseline. Epoch zero is eligible.',
        pvs_policy='All nine PVS recordings TRAIN-only; separate learned vehicle-specific ordinal boundaries; no PVS-derived numeric IRI or disturbance labels',
        test_policy='Finish and hash-freeze all selected checkpoints before this study TEST inference; no threshold/model tuning on TEST',
        source_sha256={str(p): sha(p) for p in source_files},
        limitations=['PVS quality is an input-derived proxy, not independent quality ground truth.',
                    'PVS upstream interpolated IMU has no original observation mask.',
                    'LiRA VAL has no bad sections under these fixed thresholds; bad-class validation is unavailable.',
                    'Existing TEST has prior historical exposure; this is not an unseen-city/device deployment proof.',
                    'This tests matched fine-tuning from the existing trained models, not training from scratch.'])
    write(out/'plan.json', plan)
    return plan


def check_plan(out):
    plan = read(out/'plan.json')
    expected = dict(plan['source_sha256'])
    expected.update({v['path']: v['sha256'] for v in plan['initial_checkpoints'].values()})
    expected[str(Path(plan['data'])/'manifest.json')] = plan['manifest_sha256']
    expected[str(Path(plan['pvs'])/'manifest.json')] = plan['pvs_manifest_sha256']
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(f'Frozen experiment input changed: {path}')
    return plan


def training_inputs(plan):
    data = AccelerationSpeedDataset(plan['data'], source='real', split='train', stride=1,
                                    return_labels=True, max_cached_recordings=128)
    selected, _, _ = supervised_windows(data, 16)
    pools = dataset_pools(data, selected.indices)
    alphas = {}
    for name in ('kaggle', 'roadsens'):
        subset = AccelerationSpeedDataset(plan['data'], source='real', real_dataset=name, split='train',
                                          stride=1024, return_labels=True)
        _, counts, _ = supervised_windows(subset, 16)
        alphas[name] = balanced_alpha(counts).cuda()
    # TRAIN has one LiRA road (CPH1); reject accidental mixing of section IDs.
    roads, counts = set(), defaultdict(int)
    for i, record in enumerate(data.records):
        if record.get('dataset') != 'lira_cd':
            continue
        roads.add(record['groups'][0])
        arrays = data._open(i)
        section = arrays['roughness_section']
        known = (section >= 0) & np.isfinite(arrays['overall_iri']) & arrays['mask'].any(1)
        ids, amount = np.unique(section[known], return_counts=True)
        for k, n in zip(ids, amount):
            counts[int(k)] += int(n)
    if len(roads) != 1:
        raise ValueError('Section weighting expects the declared single LiRA TRAIN road')
    weights = torch.ones(max(counts)+1, device='cuda')
    scale = np.mean(list(counts.values()))
    for k, n in counts.items():
        weights[k] = scale/n
    return data, pools, alphas, weights


def loss_for_batch(model, batch, plan, alphas, section_weights, seed, step):
    x, mask = batch['x'].cuda(), batch['mask'].cuda()
    target = {k: v.cuda() for k, v in patch_targets(batch, 16).items()}
    x = augment(x, mask, seed=seed, step=step)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(x, mask)
    rv = target['roughness_valid'] & output['patch_valid']
    section = batch['labels']['roughness_section'][:, ::16].cuda().clamp(min=0)
    weights = section_weights[section.clamp(max=len(section_weights)-1)]
    quality = output['quality_score'].new_zeros(())
    if model.mode == 'ordinal':
        quality = ordinal_loss(output['quality_logits'], iri_classes(target['roughness']), rv, weights)
    elif rv.any():
        per_patch = F.smooth_l1_loss(output['roughness'][rv].float(), target['roughness'][rv], reduction='none')
        quality = (per_patch*weights[rv]).sum()/weights[rv].sum()
    detection = quality.new_zeros(())
    for name, weight in (('kaggle', 1.), ('roadsens', .25)):
        rows = torch.tensor([v == name for v in batch['dataset']], device='cuda')[:, None]
        dv = target['disturbance_valid'] & output['patch_valid'] & rows
        if dv.any():
            y = target['disturbance'][dv]
            ce = F.binary_cross_entropy_with_logits(output['disturbance_logit'][dv].float(), y.float(), reduction='none')
            detection = detection+weight*(focal_from_cross_entropy(ce, 2.)*alphas[name][y]).mean()/1.25
    return plan['quality_weight']*quality+detection, quality, detection


def train_run(out, plan, arm, seed, inputs, validation, pvs):
    folder = out/f'{arm}_seed{seed}'
    folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():
        if sha(folder/'best.pt') != read(folder/'complete.json')['sha256']:
            raise ValueError('Completed checkpoint changed')
        return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    saved = torch.load(plan['initial_checkpoints'][str(seed)]['path'], map_location='cpu', weights_only=True)
    mode = arm.split('_')[0]
    model = OrdinalRoadModel.from_regression(saved, mode=mode).cuda()
    if plan.get('freeze_backbone', False):
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(not name.startswith(('encoder.', 'statistics_embedding.', 'disturbance_head.')))
    base, heads = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (base if name.startswith(('encoder.', 'statistics_embedding.')) else heads).append(parameter)
    groups = [dict(params=heads, lr=plan['head_lr'])]
    if base:
        groups.append(dict(params=base, lr=plan['backbone_lr']))
    optimizer = torch.optim.AdamW(groups, weight_decay=plan['weight_decay'])
    data, pools, alphas, section_weights = inputs
    history, best = [], (-float('inf'), -float('inf'))
    baseline = evaluate(model, *validation)
    initial_f1 = baseline['disturbance']['f1']
    elapsed = time.monotonic()
    config = dict(encoder_config=model.encoder.config, model_config=dict(statistics_mode='both', mode=mode),
                  arm=arm, seed=seed, input_channels=['accel_x', 'accel_y', 'accel_z', 'speed'],
                  plan_sha256=sha(out/'plan.json'), cutpoints_m_per_km=CUTPOINTS)

    def record(epoch, metrics, losses):
        nonlocal best
        score = (metrics['sections']['macro_f1_present'], -metrics['sections']['mean_absolute_class_error'])
        eligible = metrics['disturbance']['f1'] >= initial_f1-.02
        entry = dict(epoch=epoch, val=metrics, train=losses, eligible=eligible,
                     elapsed_seconds=time.monotonic()-elapsed)
        history.append(entry)
        if eligible and score > best:
            best = score
            torch.save(dict(config=config, model_state=model.state_dict(), epoch=epoch, val=metrics), folder/'best.pt')
        write(folder/'history.json', history)
        write(out/'progress.json', dict(stage='training', arm=arm, seed=seed, epoch=epoch,
              val_quality_f1=metrics['sections']['macro_f1_present'], val_detection_f1=metrics['disturbance']['f1'],
              best_quality_f1=best[0], pid=os.getpid()))
        print(f'{arm} seed={seed} epoch={epoch} quality F1={score[0]:.4f} detector F1={metrics["disturbance"]["f1"]:.4f} eligible={eligible}', flush=True)

    record(0, baseline, {})
    for epoch in range(1, plan['epochs']+1):
        indices = sample(pools, plan['samples_per_batch'], seed, epoch, plan['steps_per_epoch'])
        loader = DataLoader(Subset(data, indices), batch_size=plan['batch_size'], shuffle=False,
                            num_workers=0, generator=torch.Generator().manual_seed(seed*100+epoch))
        pvs_loader = None
        if arm.endswith('_pvs'):
            rng = np.random.default_rng([seed, epoch, 923])
            draws = rng.integers(len(pvs), size=plan['steps_per_epoch']*plan['pvs_batch_size']).tolist()
            pvs_loader = iter(DataLoader(Subset(pvs, draws), batch_size=plan['pvs_batch_size'], shuffle=False,
                              num_workers=0, generator=torch.Generator().manual_seed(seed*100+epoch+1)))
        sums = np.zeros(4)
        model.train()
        if plan.get('freeze_backbone', False):
            model.encoder.eval()
            model.statistics_embedding.eval()
            model.disturbance_head.eval()
        for step, batch in enumerate(loader):
            global_step = (epoch-1)*plan['steps_per_epoch']+step
            optimizer.zero_grad(set_to_none=True)
            loss, qloss, dloss = loss_for_batch(model, batch, plan, alphas, section_weights, seed, global_step)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite primary loss')
            loss.backward()
            auxiliary = loss.new_zeros(())
            if pvs_loader is not None:
                pb = next(pvs_loader)
                # PVS dropout may not perturb the next paired real-data dropout draw.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(seed*100000+global_step+817)
                    px, pm = pb['x'].cuda(), pb['mask'].cuda()
                    px = augment(px, pm, seed=seed+1000, step=global_step)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        po = model(px, pm, vehicle=pb['vehicle'].cuda())
                    py, pv = pvs_patch_targets(pb)
                    auxiliary = ordinal_loss(po['pvs_logits'], py.cuda(), pv.cuda() & po['patch_valid'])
                    if auxiliary.requires_grad:
                        (plan['pvs_weight']*auxiliary).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            sums += [float(loss.detach()), float(qloss.detach()), float(dloss.detach()), float(auxiliary.detach())]
        metrics = evaluate(model, *validation)
        record(epoch, metrics, dict(zip(('total', 'quality', 'disturbance', 'pvs'), (sums/len(loader)).tolist())))
    best_saved = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    if plan.get('freeze_backbone', False):
        # Both final training state and selected state must preserve every frozen tensor.
        for name, tensor in saved['model_state'].items():
            if name.startswith(('encoder.', 'statistics_embedding.', 'disturbance_head.')):
                assert torch.equal(model.state_dict()[name].cpu(), tensor)
                assert torch.equal(best_saved['model_state'][name].cpu(), tensor)
        assert best_saved['val']['disturbance'] == baseline['disturbance']
    write(folder/'complete.json', dict(sha256=sha(folder/'best.pt'), best_epoch=best_saved['epoch'],
          val=best_saved['val'], elapsed_seconds=time.monotonic()-elapsed, optimizer_updates=plan['epochs']*plan['steps_per_epoch']))
    del model, optimizer, loader
    gc.collect(); torch.cuda.empty_cache()


def run(out):
    import fcntl
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 is required for this declared experiment')
    plan = check_plan(out)
    with (out/'runner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = training_inputs(plan)
        validation = data_loader(plan['data'], 'val')
        pvs = PVSDataset(plan['pvs'])
        for seed in plan['seeds']:
            for arm in plan['arms']:
                check_plan(out)
                train_run(out, plan, arm, seed, inputs, validation, pvs)
        frozen = {f'{arm}_seed{seed}': sha(out/f'{arm}_seed{seed}'/'best.pt')
                  for seed in plan['seeds'] for arm in plan['arms']}
        receipt = dict(checkpoints=frozen, plan_sha256=sha(out/'plan.json'), test_read_at_freeze=False)
        if (out/'frozen.json').exists() and read(out/'frozen.json') != receipt:
            raise ValueError('Frozen evaluation receipt changed')
        write(out/'frozen.json', receipt)
        # TEST is first opened here, after every run and selection is complete.
        test = data_loader(plan['data'], 'test')
        results = {}
        for name, digest in frozen.items():
            path = out/name/'best.pt'
            if sha(path) != digest:
                raise ValueError('Selected weights changed before TEST')
            saved = torch.load(path, map_location='cpu', weights_only=False)
            from road_training.instance_model import InstancePatchTST
            model = OrdinalRoadModel(InstancePatchTST(**saved['config']['encoder_config']),
                                     **saved['config']['model_config']).cuda()
            model.load_state_dict(saved['model_state'])
            scores = evaluate(model, *test)
            write(out/name/'test.json', scores)
            results[name] = dict(val=saved['val'], test=scores, best_epoch=saved['epoch'])
            del model
            torch.cuda.empty_cache()
        write(out/'results.json', results)
        write(out/'progress.json', dict(stage='complete', runs=len(results), pid=os.getpid()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--init', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--pvs', type=Path)
    parser.add_argument('--ensemble', type=Path, default=Path('models/acceleration_speed/ensemble.json'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[52, 53])
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Train only quality/annotation parameters; preserve the detector exactly')
    args = parser.parse_args()
    if args.init:
        if args.data is None or args.pvs is None:
            parser.error('--init requires --data and --pvs')
        initialize(args)
    else:
        run(args.out)


if __name__ == '__main__':
    main()
