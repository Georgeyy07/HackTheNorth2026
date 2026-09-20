"""Controlled roughness-head adaptation using qualified simulated road IRI.

Keep the encoder, statistics embedding, and disturbance head exactly fixed.
Compare equal-update real-only / 25% synthetic / 50% synthetic training.
"""
import argparse
import copy
import gc
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from road_training.dataset import RoadDataset
from road_training.experiments.instance_study import make_model
from road_training.mounting_augmentation import perturb
from road_training.train_multitask import patch_targets
from road_training.common import read, write, sha


ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / 'road_training/data_roadsens_aligned'
SYNTH = ROOT / 'road_training/data_realism_diversity_20260917/F0D1'
PARENT = ROOT / 'reports/mounting_ensemble_20260919/ensemble.json'
OUT = ROOT / 'reports/synthetic_roughness_20260919'
ARMS = {'real_only': 0., 'synthetic_25': .25, 'synthetic_50': .5}


def set_head_training(model):
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('roughness_head.'))


def non_roughness_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if not key.startswith('roughness_head.')}


def assert_frozen(model, original):
    assert all(torch.equal(value, model.state_dict()[key].detach().cpu()) for key, value in original.items())


def audit_data():
    """Verify TRAIN/VAL assets, physical families and actual IRI target support."""
    summaries = {}
    for source, root in [('real', REAL), ('synthetic', SYNTH)]:
        manifest = read(root / 'manifest.json')
        groups = {}
        for split in ('train', 'val'):
            records = [r for r in manifest['records'] if r['split'] == split and r['source'] == source
                       and (source == 'synthetic' or r.get('dataset') == 'lira_cd')]
            groups[split] = {g for r in records for g in r['groups']}
            counts, values, domains = [0, 0], [], {}
            for record in records:
                folder = root / record['path']
                for name in ('x.npy', 'mask.npy', 'time.npy', 'overall_iri.npy', 'roughness_section.npy', 'metadata.json'):
                    assert sha(folder / name) == record['file_sha256'][name]
                y, section = np.load(folder / 'overall_iri.npy'), np.load(folder / 'roughness_section.npy')
                valid = np.isfinite(y) & (section >= 0)
                assert (y[valid] >= 0).all()
                counts[0] += len(y); counts[1] += int(valid.sum()); values.extend(y[valid][::16])
                domain = record.get('observation_domain', 'lira')
                domains[domain] = domains.get(domain, 0) + 1
            summaries[source + '_' + split] = dict(records=len(records), independent_groups=len(groups[split]),
                seconds=counts[0] / 100, labeled_seconds=counts[1] / 100, domains=domains,
                iri_quantiles=np.quantile(values, [0, .1, .5, .9, 1]).tolist(),
                sampled_grade_counts=np.bincount(np.digitize(values, [2., 4., 6.]), minlength=4).tolist())
        assert not groups['train'] & groups['val']
    return summaries


def initialize():
    OUT.mkdir(exist_ok=False)
    paths = list(read(PARENT)['checkpoints'])
    plan = dict(seeds=[52, 53, 54, 55], parent_receipt=str(PARENT), parent_sha256=sha(PARENT),
                parents=dict(zip([52, 53, 54, 55], paths)), parent_hashes=read(PARENT)['checkpoints'],
                arms=ARMS, epochs=12, steps_per_epoch=64, patches_per_batch=4096, lr=1e-4, weight_decay=.01,
                precision='CUDA bfloat16 autocast', normalization='Existing masked per-window instance normalization',
                frozen='Encoder, statistics_embedding and disturbance_head; only roughness_head changes',
                training_windows='Complete non-overlapping 1024-sample TRAIN windows; one clean and one fixed augmented view',
                augmentation='TRAIN only: yaw +/-20 and tilt +/-10 degrees on 75% of windows plus existing small sensor noise',
                sample_policy='Uniform section-traversal sampling within source; synthetic quota divided equally across Kaggle/LiRA observers',
                control='Same initialization, updates, total patch batch size, LR, loss, augmentation cache, and random draws. Mixtures replace real examples; real exposure decreases with mixture fraction.',
                loss='Smooth L1 on observed whole patches contained in a known 100-m IRI section; beta=1',
                selection='Minimum equal-spatial-section LiRA VAL MAE, per member; epoch-zero baseline is eligible',
                evaluation='Freeze every arm/member before any new TEST evaluation. Evaluate fixed arms on LiRA measured IRI and Kaggle prediction distributions. No fitting to Kaggle colors or TEST.',
                synthetic_evaluation='Four held-out physical families. Report observer-specific IRI errors and paired-observer consistency; no fitting on synthetic VAL.',
                source_roots=dict(real=str(REAL), synthetic=str(SYNTH)),
                manifests={str(root / 'manifest.json'): sha(root / 'manifest.json') for root in (REAL, SYNTH)},
                implementation_sha256=sha(__file__),
                limitations=['Kaggle has no IRI ground truth: changed colors do not establish improved accuracy.',
                             'LiRA VAL/TEST contain only good/medium project grades; severe-road transfer is not validated.',
                             'Synthetic TRAIN has 36 physical road families and only four held-out families.',
                             'The effective-response F1 candidate failed its calibration gate; this uses qualified F0 current response.',
                             'TEST has historical exposure; this pilot freezes selection before further TEST scoring.',
                             'Frozen-head adaptation tests whether current features can support transfer; it does not test encoder retraining.'])
    write(OUT / 'plan.json', plan)
    write(OUT / 'data_audit.json', audit_data())
    return plan


def open_model(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = make_model(saved['config']).cuda().eval()
    model.load_state_dict(saved['model_state'])
    set_head_training(model)
    return model, saved


@torch.inference_mode()
def cache_features(model, source, split, *, augment=False, seed=0, all_patches=False):
    """Cache exact FP32 inputs to the existing head, not a new encoder variant.

    A forward pre-hook captures the actual head input. Keeping FP32 avoids
    changing the head's LayerNorm input through an extra BF16 quantization.
    """
    root = SYNTH if source == 'synthetic' else REAL
    data = RoadDataset(root, source='synthetic' if source == 'synthetic' else 'real',
                       real_dataset='all' if source == 'synthetic' else source,
                       split=split, stride=1024, return_labels=True, max_cached_recordings=128)
    lookup = {r['id']: r for r in data.records}
    loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    captured, features, targets, records, sections, spatial, domains = [], [], [], [], [], [], []
    full_predictions = []
    handle = model.roughness_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        for view in range(2 if augment else 1):
            for step, batch in enumerate(loader):
                x, mask = batch['x'].cuda(non_blocking=True), batch['mask'].cuda(non_blocking=True)
                if view:
                    x = perturb(x, mask, seed=seed, step=step, rotate=True, noise=True)
                target = {k: v.cuda(non_blocking=True) for k, v in patch_targets(batch, 16).items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(x, mask)
                raw = captured.pop()
                keep = output['patch_valid'] if all_patches else output['patch_valid'] & target['roughness_valid']
                assert not captured
                if not keep.any():
                    continue
                features.append(raw[keep].float().cpu())
                targets.append(target['roughness'][keep].float().cpu())
                full_predictions.append(output['roughness'][keep].float().cpu())
                selected = keep.cpu().numpy()
                sid = batch['labels']['roughness_section'][:, ::16].numpy()
                for i, record_id in enumerate(batch['recording_id']):
                    count = int(selected[i].sum())
                    record = lookup[record_id]
                    records.extend([record_id] * count)
                    sections.extend(sid[i][selected[i]].tolist())
                    key = record.get('parent_recording', record_id) if source == 'synthetic' else record['groups'][0]
                    spatial.extend([key] * count)
                    domains.extend([record.get('observation_domain', source)] * count)
    finally:
        handle.remove()
    meta = pd.DataFrame(dict(recording=records, section=sections, spatial=spatial, domain=domains))
    cache = dict(x=torch.cat(features).cuda(), y=torch.cat(targets).cuda(), meta=meta,
                 reference_prediction=np.concatenate([p.numpy() for p in full_predictions]))
    assert len(cache['x']) == len(meta)
    # Check that feeding cached features reproduces the native model head.
    actual = predict_head(model.roughness_head, cache)
    np.testing.assert_allclose(actual, cache['reference_prediction'], rtol=2e-3, atol=.015)
    return cache


@torch.inference_mode()
def predict_head(head, cache):
    head.eval()
    rows = []
    for start in range(0, len(cache['x']), 8192):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            value = F.softplus(head(cache['x'][start:start + 8192]).squeeze(-1).float())
        rows.append(value.cpu().numpy())
    return np.concatenate(rows)


def metrics(prediction, cache, domain=None):
    frame = cache['meta'].copy()
    frame['prediction'] = prediction
    frame['target'] = cache['y'].cpu().numpy()
    if domain:
        frame = frame[frame.domain == domain]
    traversals = frame.groupby(['spatial', 'recording', 'section'], sort=True).agg(
        prediction=('prediction', 'mean'), target=('target', 'mean')).reset_index()
    sections = traversals.groupby(['spatial', 'section'], sort=True)[['prediction', 'target']].mean()
    delta = sections.prediction - sections.target
    truth_grade = np.digitize(sections.target, [2., 4., 6.])
    pred_grade = np.digitize(sections.prediction, [2., 4., 6.])
    return dict(patches=len(frame), traversals=len(traversals), sections=len(sections),
                patch_mae=float(np.abs(frame.prediction - frame.target).mean()),
                section_mae=float(np.abs(delta).mean()), section_rmse=float(np.sqrt(np.square(delta).mean())),
                section_bias=float(delta.mean()), ordinal_accuracy=float((truth_grade == pred_grade).mean()),
                sections_detail=sections.reset_index().to_dict('records'))


class SectionPool:
    """Uniform traversal/section draws, not duration-weighted random patches."""
    def __init__(self, cache, domain=None):
        meta = cache['meta']
        selected = np.arange(len(meta)) if domain is None else np.flatnonzero(meta.domain.to_numpy() == domain)
        keys = list(zip(meta.recording.to_numpy()[selected], meta.section.to_numpy()[selected]))
        codes, unique = pd.factorize(pd.MultiIndex.from_tuples(keys))
        order = np.argsort(codes, kind='stable')
        counts = np.bincount(codes)
        self.indices = torch.tensor(selected[order], device='cuda')
        self.counts = torch.tensor(counts, device='cuda')
        self.starts = torch.tensor(np.r_[0, counts.cumsum()[:-1]], device='cuda')
        self.cache = cache

    def sample(self, count, generator):
        groups = torch.randint(len(self.counts), (count,), device='cuda', generator=generator)
        within = (torch.rand(count, device='cuda', generator=generator) * self.counts[groups]).long()
        indices = self.indices[self.starts[groups] + within]
        return self.cache['x'][indices], self.cache['y'][indices]


def save_checkpoint(path, model, saved, parent, epoch, arm, validation):
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    # Parent joint metrics describe the old head; never present them as the
    # adapted model's validation scores. This study has a separate IRI protocol.
    payload = {key: value for key, value in saved.items() if key != 'val_metrics'}
    torch.save(dict(payload, model_state=state, parent_val_metrics=saved.get('val_metrics'),
                   roughness_validation=validation,
                   roughness_adaptation=dict(arm=arm, epoch=epoch,
                   parent_sha256=sha(parent), plan_sha256=sha(OUT / 'plan.json'))), path)


def train_member(seed, parent, plan):
    model, saved = open_model(parent)
    frozen = non_roughness_state(model)
    initial = copy.deepcopy(model.roughness_head.state_dict())
    start = time.monotonic()
    print(f'seed {seed}: caching clean + augmented TRAIN features', flush=True)
    real = cache_features(model, 'lira', 'train', augment=True, seed=seed)
    synthetic = cache_features(model, 'synthetic', 'train', augment=True, seed=seed)
    validation = cache_features(model, 'lira', 'val')
    synthetic_val = cache_features(model, 'synthetic', 'val')
    pools = [SectionPool(real), SectionPool(synthetic, 'kaggle'), SectionPool(synthetic, 'lira')]
    original_pred = predict_head(model.roughness_head, validation)
    original_metric = metrics(original_pred, validation)
    summaries = {}
    for arm, fraction in plan['arms'].items():
        folder = OUT / f'seed{seed}' / arm
        folder.mkdir(parents=True, exist_ok=False)
        model.roughness_head.load_state_dict(initial)
        torch.manual_seed(seed)
        optimizer = torch.optim.AdamW(model.roughness_head.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
        generator = torch.Generator(device='cuda').manual_seed(seed + 190919)
        best, epoch_best = original_metric['section_mae'], 0
        history = [dict(epoch=0, validation={k: v for k, v in original_metric.items() if k != 'sections_detail'})]
        save_checkpoint(folder / 'best.pt', model, saved, parent, 0, arm, original_metric)
        total = plan['patches_per_batch']
        count_syn = int(total * fraction / 2)
        count_real = total - 2 * count_syn
        for epoch in range(1, plan['epochs'] + 1):
            model.roughness_head.train()
            losses = []
            for step in range(plan['steps_per_epoch']):
                # All arms draw the same candidate indices; only quotas differ.
                xr, yr = pools[0].sample(total, generator)
                xk, yk = pools[1].sample(total // 2, generator)
                xl, yl = pools[2].sample(total // 2, generator)
                x = torch.cat((xr[:count_real], xk[:count_syn], xl[:count_syn]))
                y = torch.cat((yr[:count_real], yk[:count_syn], yl[:count_syn]))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = F.softplus(model.roughness_head(x).squeeze(-1).float())
                    loss = F.smooth_l1_loss(prediction, y, beta=1.)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Invalid head loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.roughness_head.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); losses.append(float(loss.detach()))
            score = metrics(predict_head(model.roughness_head, validation), validation)
            history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)),
                                validation={k: v for k, v in score.items() if k != 'sections_detail'}))
            if score['section_mae'] < best:
                best = score['section_mae']; epoch_best = epoch
                save_checkpoint(folder / 'best.pt', model, saved, parent, epoch, arm, score)
            write(folder / 'history.json', history)
            print(f'seed={seed} {arm} epoch={epoch} VAL section MAE={score["section_mae"]:.4f} best={best:.4f}', flush=True)
        checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        assert_frozen(model, frozen)
        real_metrics = metrics(predict_head(model.roughness_head, validation), validation)
        syn_prediction = predict_head(model.roughness_head, synthetic_val)
        syn_metrics = {domain: metrics(syn_prediction, synthetic_val, domain) for domain in ('kaggle', 'lira')}
        summary = dict(seed=seed, arm=arm, best_epoch=epoch_best, parent_section_mae=original_metric['section_mae'],
                       lira_validation=real_metrics, synthetic_validation=syn_metrics,
                       checkpoint=str(folder / 'best.pt'), checkpoint_sha256=sha(folder / 'best.pt'),
                       unchanged_non_roughness_tensors=len(frozen))
        write(folder / 'complete.json', summary); summaries[arm] = summary
        del optimizer
    write(OUT / f'seed{seed}/complete.json', dict(arms=summaries, elapsed_s=time.monotonic() - start))
    del model, real, synthetic, validation, synthetic_val, pools
    gc.collect(); torch.cuda.empty_cache()


def distributions(values):
    return dict(patches=len(values), mean=float(values.mean()), median=float(np.median(values)),
                quantiles=np.quantile(values, [.05, .25, .5, .75, .95]).tolist(),
                grade_percent=(np.bincount(np.digitize(values, [2., 4., 6.]), minlength=4) / len(values) * 100).tolist())


def evaluate_frozen(plan, split):
    """All arms get the same clean inputs; TEST never affects checkpoint choice."""
    predictions = {arm: {} for arm in ['baseline', *ARMS]}
    last_cache = {}
    for seed in plan['seeds']:
        parent = plan['parents'][str(seed)]
        model, saved = open_model(parent)
        original = non_roughness_state(model)
        for source in ('lira', 'kaggle', *(['synthetic'] if split == 'val' else [])):
            cache = cache_features(model, source, split, all_patches=source == 'kaggle')
            predictions['baseline'].setdefault(source, []).append(predict_head(model.roughness_head, cache))
            for arm in ARMS:
                folder = OUT / f'seed{seed}' / arm
                checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
                head = {key.removeprefix('roughness_head.'): value for key, value in checkpoint['model_state'].items() if key.startswith('roughness_head.')}
                model.roughness_head.load_state_dict(head)
                assert_frozen(model, original)
                predictions[arm].setdefault(source, []).append(predict_head(model.roughness_head, cache))
            model.load_state_dict(saved['model_state'])
            last_cache[source] = dict(meta=cache['meta'], y=cache['y'].cpu())
            del cache
        del model; gc.collect(); torch.cuda.empty_cache()
    summary = {}
    arrays = {}
    for arm, by_source in predictions.items():
        summary[arm] = {}
        for source, member_values in by_source.items():
            values = np.mean(member_values, axis=0)
            arrays[arm + '__' + source] = values
            if source == 'kaggle':
                summary[arm]['kaggle_prediction_distribution'] = distributions(values)
            elif source == 'synthetic':
                summary[arm]['synthetic'] = {domain: metrics(values, last_cache[source], domain) for domain in ('kaggle', 'lira')}
                frame = last_cache[source]['meta'].copy(); frame['prediction'] = values
                # Average over each physical traversal/section within each device view.
                paired = frame.groupby(['spatial', 'section', 'domain']).prediction.mean().unstack('domain').dropna()
                summary[arm]['paired_observer_iri_gap'] = float(np.abs(paired.kaggle - paired.lira).mean())
            else:
                summary[arm]['lira'] = metrics(values, last_cache[source])
    np.savez_compressed(OUT / f'{split}_predictions.npz', **arrays)
    for source, cache in last_cache.items():
        cache['meta'].assign(target=cache['y'].numpy()).to_parquet(OUT / f'{split}_{source}_rows.parquet', index=False)
    write(OUT / f'{split}_results.json', summary)
    return summary


def run():
    if not (OUT / 'plan.json').exists():
        initialize()
    plan = read(OUT / 'plan.json')
    assert sha(__file__) == plan['implementation_sha256']
    for path, digest in plan['manifests'].items():
        assert sha(path) == digest
    for path, digest in plan['parent_hashes'].items():
        assert sha(path) == digest
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    for seed in plan['seeds']:
        write(OUT / 'progress.json', dict(stage='training', seed=seed, pid=os.getpid()))
        if not (OUT / f'seed{seed}/complete.json').exists():
            train_member(seed, plan['parents'][str(seed)], plan)
    # Freeze receipts before computing any new TEST prediction.
    for arm in ARMS:
        receipt = dict(seeds=plan['seeds'], checkpoints={str(OUT / f'seed{s}' / arm / 'best.pt'): sha(OUT / f'seed{s}' / arm / 'best.pt') for s in plan['seeds']},
                       aggregation='Equal mean of four probabilities and four IRI predictions',
                       parent_receipt_sha256=sha(PARENT), plan_sha256=sha(OUT / 'plan.json'), test_read_at_freeze=False)
        write(OUT / f'{arm}_ensemble.json', receipt)
    write(OUT / 'progress.json', dict(stage='validation', pid=os.getpid()))
    validation = evaluate_frozen(plan, 'val')
    winner = min(validation, key=lambda arm: validation[arm]['lira']['section_mae'])
    synthetic_winner = min([arm for arm in ARMS if arm != 'real_only'], key=lambda arm: validation[arm]['lira']['section_mae'])
    write(OUT / 'selection.json', dict(winner=winner, best_synthetic=synthetic_winner,
        criterion='LiRA VAL section MAE only; Kaggle prediction distribution not used', test_read_at_selection=False,
        receipts={arm: sha(OUT / f'{arm}_ensemble.json') for arm in ARMS}))
    write(OUT / 'progress.json', dict(stage='test_evaluation', pid=os.getpid()))
    evaluate_frozen(plan, 'test')
    write(OUT / 'progress.json', dict(stage='complete', winner=winner, best_synthetic=synthetic_winner))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', action='store_true')
    args = parser.parse_args()
    if args.init:
        initialize()
    else:
        run()
