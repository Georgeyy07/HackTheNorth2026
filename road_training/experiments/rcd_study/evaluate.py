"""Freeze selected weights, then score VAL and one fixed final TEST comparison."""
import argparse
from datetime import datetime, timezone

import torch

from road_training.rcd import RCDEncoder
from road_training.patchtst import PatchTST, PatchTSTRoadModel
from road_training.common import read, write, sha
from road_training.metrics import evaluate as event_metrics
from road_training.domain_evaluation import evaluate as patch_metrics
from road_training.experiments.rcd_study.study import OUT, check_plan
from road_training.pretraining_adaptation import FrozenLinearRoadModel


def evaluate_split(split):
    plan = check_plan()
    if split == 'test':
        frozen = read(OUT/'frozen_checkpoints.json')
        if frozen['plan_sha256'] != sha(OUT/'plan.json'):
            raise ValueError('Plan changed after freezing')
        for row in frozen['rows']:
            if sha(row['checkpoint']) != row['sha256']:
                raise ValueError('Checkpoint changed after freezing')
    for arm, setting in plan['arms'].items():
        kind = RCDEncoder
        model_class = FrozenLinearRoadModel if setting.get('adaptation')=='linear_probe' else PatchTSTRoadModel
        for seed in plan['seeds']:
            folder = OUT/f'{arm}_seed{seed}'
            if read(folder/'progress.json')['state'] != 'complete':
                raise ValueError('Finish all training before held-out comparison')
            patch_metrics(folder/'best.pt', folder/f'patch_{split}.json', source='real', split=split, encoder_class=kind, model_class=model_class)
            event_metrics(folder/'best.pt', folder/f'event_section_{split}.json', split=split, encoder_class=kind, model_class=model_class)
            if split == 'val':
                selected = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)['val_metrics']
                reproduced = read(folder/'patch_val.json')['metrics']
                old_f1 = selected['disturbance']['classification']['per_class']['disturbance']['f1']
                new_f1 = reproduced['disturbance']['classification']['per_class']['disturbance']['f1']
                if old_f1 != new_f1 or abs(selected['roughness']['mae']-reproduced['roughness']['mae']) > 1e-8:
                    raise ValueError('Standalone VAL evaluation does not reproduce checkpoint metrics')
            print(f'{split.upper()} evaluation complete: {arm}, seed {seed}', flush=True)
    write(OUT/f'{split}_evaluation_complete.json', dict(passed=True, split=split,
        plan_sha256=sha(OUT/'plan.json'), completed_utc=datetime.now(timezone.utc).isoformat()))


def freeze():
    plan = check_plan()
    if not read(OUT/'val_evaluation_complete.json')['passed']:
        raise ValueError('Finish and verify VAL first')
    path = OUT/'frozen_checkpoints.json'
    if path.exists():
        raise FileExistsError('Do not replace the frozen comparison')
    rows = []
    for arm in plan['arms']:
        for seed in plan['seeds']:
            folder = OUT/f'{arm}_seed{seed}'
            history = read(folder/'history.json')
            best = min(history, key=lambda r:r['val']['loss'])
            saved = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
            if best['updates'] != saved['updates']:
                raise ValueError('Checkpoint selection differs from frozen rule')
            setting = plan['arms'][arm]
            if setting.get('adaptation')=='linear_probe':
                parent = torch.load(OUT/f"{setting['method']}_pretrain_seed{seed}"/'selected.pt',
                                    map_location='cpu', weights_only=True)['encoder_state']
                for key,value in parent.items():
                    if not torch.equal(value,saved['model_state']['encoder.'+key]):
                        raise ValueError('Linear probe changed its pretrained encoder')
            if setting.get('adaptation')=='low_lr':
                groups = saved['config']['optimizer_groups']
                if groups[0]['lr'] != 5e-6 or groups[1]['lr'] != 1e-4:
                    raise ValueError('Encoder/head learning rates do not match the requested ratio')
            rows.append(dict(arm=arm, seed=seed, checkpoint=str(folder/'best.pt'),
                sha256=sha(folder/'best.pt'), config_sha256=sha(folder/'config.json'),
                history_sha256=sha(folder/'history.json'), selected_updates=best['updates']))
    for method, arms in [('rcd', ('rcd_scratch','rcd_finetuned'))]:
        for seed in plan['seeds']:
            control = OUT/f'{arms[1]}_seed{seed}'
            reference = read(control/'config.json')
            for arm in [arms[0], f'{method}_linear', f'{method}_low_lr']:
                folder = OUT/f'{arm}_seed{seed}'
                config = read(folder/'config.json')
                keys = ['encoder_config', 'train_statistics', 'loss']
                if config.get('adaptation') != 'linear_probe':
                    keys.append('head_initialization_sha256')
                for key in keys:
                    if config[key] != reference[key]:
                        raise ValueError(f'Paired control mismatch: {arm} {key}')
                for a, b in zip(read(folder/'history.json'), read(control/'history.json')):
                    if a['sampled_prefix_sha256'] != b['sampled_prefix_sha256']:
                        raise ValueError('Paired fine-tuning windows differ')
    write(path, dict(frozen_utc=datetime.now(timezone.utc).isoformat(), rows=rows,
        plan_sha256=sha(OUT/'plan.json'), manifest_sha256=plan['manifest_sha256'],
        threshold=.5, decision='Evaluate every planned arm; no TEST-dependent selection or reruns',
        historical_test_exposure=True, paired_initialization_and_sampling_verified=True))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['val', 'freeze', 'test'])
    args=p.parse_args()
    if args.action == 'freeze': freeze()
    else: evaluate_split(args.action)
