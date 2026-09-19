"""Same real-label adaptation protocol as the previous pretraining comparison."""
import argparse
import hashlib
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from road_training.common import read, write, sha
from road_training.sampling import sample_indices
from road_training.dataset import RoadDataset
from road_training.patchtst import PatchTSTRoadModel
from road_training.rcd import RCDEncoder, load_rcd_encoder
from road_training.train import balanced_alpha
from road_training.train_multitask import run_epoch, supervised_windows
from road_training.block_sampling import CurrentIndices
from road_training.pretraining_adaptation import FrozenLinearRoadModel, make_optimizer
from road_training.experiments.rcd_study.study import OUT, CORPUS, ARMS, check_plan


def fine_tune(arm, seed):
    plan = check_plan()
    if arm not in plan['arms'] or seed not in plan['seeds']:
        raise ValueError('Unknown frozen arm or seed')
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required')
    setting = plan['arms'][arm]
    parent = OUT / f"{setting['method']}_pretrain_seed{seed}"
    if read(parent / 'progress.json')['state'] != 'complete':
        raise ValueError('Complete the paired pretraining run first')
    preconfig = read(parent / 'config.json')
    if setting['pretrained']:
        encoder = load_rcd_encoder(parent / 'selected.pt')
    else:
        encoder = RCDEncoder(**preconfig['encoder_config'], train_stats=preconfig['train_statistics'])
    # Decoder construction/loading cannot change either the new head weights or
    # subsequent dropout sequence in a paired pretrained/scratch comparison.
    torch.manual_seed(seed + 10000)
    adaptation = setting.get('adaptation', 'full')
    model_class = FrozenLinearRoadModel if adaptation == 'linear_probe' else PatchTSTRoadModel
    model = model_class(encoder).cuda()
    initial_encoder = {name:p.detach().cpu().clone() for name,p in encoder.named_parameters()}
    head_digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not name.startswith('encoder.'):
            head_digest.update(value.cpu().numpy().tobytes())
    torch.manual_seed(seed + 20000)
    data = RoadDataset(CORPUS, source='real', split='train', stride=1, return_labels=True)
    selected, counts, _ = supervised_windows(data, 16)
    pool = np.asarray(selected.indices, np.int64)
    val = RoadDataset(CORPUS, source='real', split='val', stride=1024, return_labels=True)
    val_selected, _, _ = supervised_windows(val, 16)
    alpha = balanced_alpha(counts).cuda()
    loss_args = dict(precision='bf16', gamma=2., alpha=alpha, roughness_weight=1., disturbance_weight=1.)
    optimizer = make_optimizer(model, adaptation)
    output = OUT / f'{arm}_seed{seed}'
    output.mkdir(exist_ok=False)
    current = CurrentIndices()
    loader_args = dict(batch_size=256, num_workers=2, pin_memory=True, persistent_workers=True)
    loader = DataLoader(data, sampler=current, generator=torch.Generator().manual_seed(seed+700), **loader_args)
    val_loader = DataLoader(val_selected, shuffle=False,
        generator=torch.Generator().manual_seed(seed+701), **loader_args)
    probe = sample_indices(pool, np.empty(0, np.int64), seed=916, epoch=1,
                           steps=4, batch_size=256, synthetic_per_batch=0)
    probe_loader = DataLoader(Subset(data, probe.tolist()), shuffle=False,
        generator=torch.Generator().manual_seed(seed+702), **loader_args)
    config = dict(arm=arm, seed=seed, encoder_kind=setting['method'], data_root=str(CORPUS),
        manifest_sha256=plan['manifest_sha256'], encoder_config=encoder.config,
        train_statistics=preconfig['train_statistics'], plan_sha256=sha(OUT/'plan.json'),
        parent_checkpoint=str(parent/'selected.pt') if setting['pretrained'] else None,
        parent_checkpoint_sha256=sha(parent/'selected.pt') if setting['pretrained'] else None,
        head_initialization_sha256=head_digest.hexdigest(),
        parameters=sum(p.numel() for p in model.parameters()), arguments=plan['fine_tuning'],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        adaptation=adaptation, head_type='linear' if adaptation=='linear_probe' else 'mlp',
        optimizer_groups=[dict(name=g.get('name','all'),lr=g['lr'],weight_decay=g['weight_decay'],
            parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups],
        loss=dict(gamma=2., alpha=alpha.cpu().tolist(), roughness_weight=1., disturbance_weight=1.),
        train_windows=len(selected), val_windows=len(val_selected), source='real')
    write(output/'config.json', config)
    best = significant_best = float('inf')
    stale = 0; history = []; started = time.monotonic()
    actual_updates = [0]
    hook = optimizer.register_step_post_hook(lambda *_: actual_updates.__setitem__(0, actual_updates[0]+1))
    sampling_digest = hashlib.sha256()
    recipe = plan['fine_tuning']
    for block in range(recipe['max_updates']//recipe['validation_every']):
        updates = (block+1) * recipe['validation_every']
        indices = sample_indices(pool, np.empty(0, np.int64), seed=seed, epoch=block+1,
            steps=recipe['validation_every'], batch_size=256, synthetic_per_batch=0)
        sampling_digest.update(indices.tobytes())
        current.indices = indices.tolist()
        training = run_epoch(model, loader, 'cuda', optimizer, **loss_args)
        if actual_updates[0] != updates:
            raise RuntimeError('Skipped eligible supervised batch')
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            validation = run_epoch(model, val_loader, 'cuda', **loss_args)
        value = validation['loss']
        if value < significant_best - recipe['minimum_improvement']:
            significant_best = value; stale = 0
        else:
            stale += 1
        row = dict(updates=updates, train=training, val=validation,
                   sampled_prefix_sha256=sampling_digest.hexdigest(), elapsed_seconds=time.monotonic()-started)
        history.append(row)
        saved = dict(epoch=updates/2048, updates=updates, config=config,
                     model_state=model.state_dict(), val_metrics=validation)
        if value < best:
            best = value; torch.save(saved, output/'best.pt')
        write(output/'history.json', history)
        write(output/'progress.json', dict(state='running', updates=updates, best_loss=best, stale=stale))
        f1 = validation['disturbance']['classification']['per_class']['disturbance']['f1']
        print(f'{arm} seed={seed} updates={updates} VAL loss={value:.5f} F1={f1:.4f} IRI MAE={validation["roughness"]["mae"]:.4f}', flush=True)
        if updates >= recipe['minimum_updates'] and stale >= recipe['patience_checks']:
            break
    torch.save(saved, output/'last.pt')
    def encoder_change():
        squared = total = 0.
        unchanged = True
        for name,p in encoder.named_parameters():
            now = p.detach().cpu(); before = initial_encoder[name]
            unchanged &= torch.equal(now, before)
            squared += float((now.double()-before.double()).square().sum())
            total += float(before.double().square().sum())
        return dict(bitwise_unchanged=unchanged, relative_l2_change=(squared/max(total,1e-30))**.5)
    change_last = encoder_change()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        write(output/'clean_train_last.json', run_epoch(model, probe_loader, 'cuda', **loss_args))
        best_saved = torch.load(output/'best.pt', map_location='cuda', weights_only=True)
        model.load_state_dict(best_saved['model_state'])
        write(output/'clean_train_best.json', run_epoch(model, probe_loader, 'cuda', **loss_args))
    change_best = encoder_change()
    if adaptation == 'linear_probe' and not (change_last['bitwise_unchanged'] and change_best['bitwise_unchanged']):
        raise RuntimeError('Frozen probe encoder changed')
    write(output/'encoder_change.json', dict(adaptation=adaptation, best=change_best, last=change_last))
    hook.remove()
    write(output/'progress.json', dict(state='complete', updates=updates, best_updates=best_saved['updates'],
        best_loss=best, elapsed_seconds=time.monotonic()-started, test_evaluated=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, required=True)
    a=p.parse_args();fine_tune(a.arm,a.seed)
