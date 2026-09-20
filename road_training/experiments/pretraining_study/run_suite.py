"""Run declared jobs serially, detached from the conversational process."""
import subprocess
import sys
import time
import shutil

import torch

from road_training.common import ROOT, read, write, sha
from road_training.experiments.pretraining_study.study import OUT, CORPUS, check_plan


def run_job(name, command, completed):
    write(OUT/'progress.json', dict(state='running', active=name, completed=completed))
    started = time.monotonic()
    with (OUT/f'{name}.log').open('ab', buffering=0) as log:
        result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        write(OUT/'progress.json', dict(state='failed', active=name, completed=completed, returncode=result.returncode))
        raise RuntimeError(f'{name} failed; inspect its log')
    completed.append(dict(job=name, seconds=time.monotonic()-started))


def main():
    plan = check_plan(); completed = []
    for seed in plan['seeds']:
        for method in ['droppatch', 'arctan']:
            name = f'{method}_pretrain_seed{seed}'
            folder = OUT/name
            if not (folder/'progress.json').exists():
                recipe = plan['pretraining']
                stop_epoch = recipe['method_epochs'][method]
                command = [sys.executable, '-m', 'road_training.pretrain', '--method', method,
                    '--data-root', str(CORPUS), '--source', recipe['source'], '--val-source', recipe['val_source'],
                    '--epochs', str(recipe['epochs']), '--steps-per-epoch', str(recipe['steps_per_epoch']),
                    '--val-windows', str(recipe['val_windows']), '--batch-size', str(recipe['batch_size']),
                    '--lr', str(recipe['lr']), '--workers', '2', '--seed', str(seed), '--output', str(folder),
                    '--stop-after-epochs', str(stop_epoch)]
                run_job(name, command, completed)
                history = read(folder/'history.json')
                if len(history) != stop_epoch or sum(r['train']['optimizer_updates'] for r in history) != stop_epoch*recipe['steps_per_epoch']:
                    raise ValueError('Pretraining update budget was not completed')
                write(folder/'progress.json', dict(state='complete', epochs=len(history),
                    updates=sum(r['train']['optimizer_updates'] for r in history),
                    best_checkpoint_sha256=sha(folder/'best.pt')))
            else:
                if read(folder/'progress.json')['state'] != 'complete':
                    raise ValueError('Partial pretraining run requires explicit recovery')
                completed.append(dict(job=name, reused_completed=True))
            if method == 'droppatch':
                # Seed 42 had already finished 50 epochs when the user chose
                # epoch 12; its best checkpoint is exactly that saved epoch.
                candidates = [folder/'last.pt', folder/'best.pt']
                selected = next((p for p in candidates if
                    torch.load(p,map_location='cpu',weights_only=True)['epoch']==12), None)
                if selected is None:
                    raise ValueError('The requested DropPatch epoch-12 weights are unavailable')
            else:
                selected = folder/'best.pt'
            shutil.copy2(selected, folder/'selected.pt')
            write(folder/'selection.json', dict(checkpoint=str(folder/'selected.pt'),
                sha256=sha(folder/'selected.pt'), source_checkpoint=str(selected),
                epoch=torch.load(selected,map_location='cpu',weights_only=True)['epoch'],
                rule='Fixed epoch 12 per user request' if method=='droppatch' else 'Minimum VAL reconstruction loss'))
            arms = [arm for arm,setting in plan['arms'].items() if setting['method']==method]
            for arm in arms:
                name = f'{arm}_seed{seed}'
                progress = OUT/name/'progress.json'
                if progress.exists() and read(progress)['state']=='complete':
                    completed.append(dict(job=name, reused_completed=True)); continue
                run_job(name, [sys.executable, '-m', 'road_training.experiments.pretraining_study.study',
                              '--arm', arm, '--seed', str(seed)], completed)
    write(OUT/'training_complete.json', dict(passed=True, jobs=completed, plan_sha256=sha(OUT/'plan.json')))
    for action in ['val','freeze','test']:
        run_job('evaluate_'+action, [sys.executable, '-m', 'road_training.experiments.pretraining_study.evaluate', action], completed)
    run_job('report', [sys.executable, '-m', 'road_training.experiments.pretraining_study.report'], completed)
    write(OUT/'progress.json', dict(state='complete', completed=completed, plan_sha256=sha(OUT/'plan.json')))


if __name__ == '__main__': main()
