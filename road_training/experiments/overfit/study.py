"""Paired, validation-only ablations; old datasets/checkpoints stay unchanged."""
import argparse
import gc
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from road_training.dataset import RoadDataset
from road_training.patchtst import PatchTST, PatchTSTRoadModel
from road_training.train import balanced_alpha
from road_training.train_multitask import run_epoch, supervised_windows
from road_training.common import ROOT, DATA, read, write, sha
from road_training.sampling import pools, sample_indices
from road_training.augmentation import AugmentedModel
from road_training.block_sampling import BlockSampler, CurrentIndices

OUT = ROOT / 'reports/overfit_20260916'
CORPUS = DATA.parent / 'data_transfer_v2_mount'
ARMS = {
    'M_base': {},
    'M_blocks': dict(blocks=True),
    'M_small': dict(small=True),
    'M_regularized': dict(regularized=True),
    'M_rotation': dict(rotate=True),
    'M_noise': dict(noise=True),
    'M_combined': dict(blocks=True, small=True, regularized=True, rotate=True, noise=True),
    'R_base': dict(real_only=True),
    'R_combined': dict(real_only=True, blocks=True, small=True, regularized=True, rotate=True, noise=True),
}


def write_plan():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'plan.json').exists():
        raise FileExistsError('Preserve the frozen study plan')
    sources = [Path(__file__), Path(__file__).resolve().parents[3] / 'road_training/block_sampling.py', Path(__file__).resolve().parents[3] / 'road_training/augmentation.py',
               ROOT/'road_training/train_multitask.py', ROOT/'road_training/patchtst.py', ROOT/'road_training/dataset.py']
    write(OUT / 'plan.json', dict(arms=ARMS, seeds=[42, 43, 44], max_updates=12288, validation_every=256,
        early_stopping=dict(minimum_updates=1024, patience_checks=6, minimum_improvement=1e-4),
        batch_size=256, lr=1e-4, precision='bf16', device='cuda', synthetic_per_mixed_batch=41,
        selection='Minimum observed combined real-validation loss; threshold fixed at 0.5',
        regularized=dict(encoder_and_head_dropout=.25, weight_decay=.05),
        small=dict(d_model=64, n_layers=2, n_heads=4, ffn_dim=256),
        augmentation=dict(rotation_degrees=5.,accel_jitter_mps2=.005,gyro_jitter_radps=.0002,
                          accel_bias_bound_mps2=.02,gyro_bias_bound_radps=.001,
                          noise='Clipped at 3 SD; held identical vectors share jitter; bias constant per window'),
        controls='Single-factor mixed arms plus fixed combined arms; baseline and all arms get identical validation frequency/stopping. Task/source/device slots paired; block balancing changes only within-stratum selection.',
        limitations=['No new independent roads from augmentation or block sampling.',
                     'Kaggle block distance is integrated valid speed, not map-matched road identity.',
                     'Unknown window centres group by nearest supported section/station inside that window; no target imputation or temporal pseudo-blocks.',
                     'No time/magnitude warp, signal inversion, label interpolation or generated gyro.',
                     'Real validation is development data; no TEST reads or final-generalization claim.'],
        data_root=str(CORPUS), manifest_sha256=sha(CORPUS/'manifest.json'),
        source_sha256={str(p):sha(p) for p in sources}))


def train(arm, seed):
    plan = read(OUT / 'plan.json')
    if arm not in plan['arms'] or seed not in plan['seeds']:
        raise ValueError('Run is not in the frozen plan')
    for path, expected in plan['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Study implementation changed: {path}')
    if sha(CORPUS/'manifest.json') != plan['manifest_sha256']:
        raise ValueError('Corpus changed')
    if shutil.disk_usage(ROOT).free < 1.5 * 1024**3:
        raise OSError('Less than 1.5 GiB available; stopping before new checkpoints')
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 is required')
    setting = plan['arms'][arm]
    source = 'real' if setting.get('real_only') else 'both'
    data = RoadDataset(CORPUS, source=source, split='train', stride=1, return_labels=True)
    selected, _, _ = supervised_windows(data, 16)
    real, synthetic = pools(data, selected)
    blocks = BlockSampler(data, selected.indices) if setting.get('blocks') else None
    val = RoadDataset(CORPUS, source='real', split='val', stride=1024, return_labels=True)
    val_selected, _, _ = supervised_windows(val, 16)
    alpha_data = RoadDataset(DATA, source=source, split='train', stride=1024, return_labels=True)
    _, counts, _ = supervised_windows(alpha_data, 16)
    alpha = balanced_alpha(counts).to('cuda')
    ns = 0 if source == 'real' else 41
    small, regularized = setting.get('small', False), setting.get('regularized', False)
    dropout = .25 if regularized else .1
    encoder = PatchTST(d_model=64 if small else 128, n_layers=2 if small else 3,
                      ffn_dim=256 if small else 512, dropout=dropout,
                      patch_length=16, max_patches=64, train_stats=data.train_stats)
    base = PatchTSTRoadModel(encoder, dropout=dropout).to('cuda')
    model = AugmentedModel(base, seed, setting.get('rotate', False), setting.get('noise', False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.05 if regularized else .01)
    loss_args = dict(precision='bf16', gamma=2., alpha=alpha, roughness_weight=1., disturbance_weight=1.)
    output = OUT / f'{arm}_seed{seed}'
    output.mkdir(exist_ok=False)
    index = CurrentIndices()
    loader_args = dict(batch_size=256, num_workers=2, pin_memory=True, persistent_workers=True)
    loader = DataLoader(data, sampler=index, generator=torch.Generator().manual_seed(seed+700), **loader_args)
    val_loader = DataLoader(val_selected, shuffle=False, generator=torch.Generator().manual_seed(seed+701), **loader_args)
    probe_indices = sample_indices(real, synthetic, seed=916, epoch=1, steps=4, batch_size=256, synthetic_per_batch=ns)
    probe_loader = DataLoader(Subset(data, probe_indices.tolist()), shuffle=False,
                             generator=torch.Generator().manual_seed(seed+702), **loader_args)
    config = dict(arm=arm, seed=seed, data_root=str(CORPUS), manifest_sha256=plan['manifest_sha256'],
        encoder_config=encoder.config, head_dropout=dropout, train_statistics=data.train_stats,
        parameters=sum(p.numel() for p in base.parameters()), plan_sha256=sha(OUT/'plan.json'),
        setting=setting, arguments=dict(lr=1e-4,batch_size=256,precision='bf16',device='cuda',
            weight_decay=.05 if regularized else .01),
        loss=dict(gamma=2.,alpha=alpha.cpu().tolist(),roughness_weight=1.,disturbance_weight=1.),
        source='real' if ns==0 else 'both', selection_metric='val.loss',
        probe='Fixed 1024 clean TRAIN windows; evaluation mode; no augmentation; source slots match training')
    write(output/'config.json', config)
    if blocks:
        write(output/'sampling_groups.json', dict(records=blocks.summary, policy=BlockSampler.__doc__))
    best = float('inf'); significant_best = float('inf'); stale = 0; history = []
    early = plan['early_stopping']; start_time = time.monotonic()
    optimizer_steps = [0]
    handle = optimizer.register_step_post_hook(lambda *_: optimizer_steps.__setitem__(0, optimizer_steps[0]+1))
    for epoch in range(1, 7):
        baseline = sample_indices(real, synthetic, seed=seed, epoch=epoch, steps=2048, batch_size=256, synthetic_per_batch=ns)
        indices = blocks.sample(baseline, seed * 100 + epoch) if blocks else baseline
        for offset in range(0, 2048, plan['validation_every']):
            updates = (epoch-1)*2048 + offset + plan['validation_every']
            index.indices = indices[offset*256:(offset+plan['validation_every'])*256].tolist()
            training = run_epoch(model, loader, 'cuda', optimizer, **loss_args)
            if optimizer_steps[0] != updates:
                raise RuntimeError('An eligible batch skipped supervision; update budget is not exact')
            # Evaluation and worker seeds must not consume the dropout RNG.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                validation = run_epoch(model, val_loader, 'cuda', **loss_args)
            value = validation['loss']
            if value < significant_best - early['minimum_improvement']:
                significant_best = value; stale = 0
            else:
                stale += 1
            row = dict(updates=updates, epoch=updates/2048, train=training, val=validation,
                       elapsed_seconds=time.monotonic()-start_time)
            history.append(row)
            saved = dict(epoch=updates/2048, updates=updates, config=config, model_state=base.state_dict(), val_metrics=validation)
            if value < best:
                best = value; torch.save(saved, output/'best.pt')
            write(output/'history.json', history)
            write(output/'progress.json',dict(state='running',updates=updates,checks=len(history),best_loss=best,stale=stale))
            f1 = validation['disturbance']['classification']['per_class']['disturbance']['f1']
            print(f'{arm} seed={seed} updates={updates} val_loss={value:.5f} F1={f1:.4f} IRI={validation["roughness"]["mae"]:.4f}', flush=True)
            if updates >= early['minimum_updates'] and stale >= early['patience_checks']:
                break
        if updates >= early['minimum_updates'] and stale >= early['patience_checks']:
            break
    torch.save(saved, output/'last.pt')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        write(output/'clean_train_last.json', run_epoch(model, probe_loader, 'cuda', **loss_args))
        saved_best = torch.load(output/'best.pt', map_location='cuda', weights_only=False)
        base.load_state_dict(saved_best['model_state'])
        write(output/'clean_train_best.json', run_epoch(model, probe_loader, 'cuda', **loss_args))
    handle.remove()
    write(output/'progress.json',dict(state='complete',updates=updates,best_updates=saved_best['updates'],
        early_stopped=updates<plan['max_updates'],best_loss=best,test_evaluated=False,
        elapsed_seconds=time.monotonic()-start_time))
    del loader, val_loader, probe_loader, model, base, optimizer
    gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',action='store_true');p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int)
    args=p.parse_args()
    if args.plan:write_plan()
    elif args.arm and args.seed is not None:train(args.arm,args.seed)
    else:p.error('Use --plan, or both --arm and --seed')
