"""Serial detached runner, with explicit completion receipts for each job."""
import shutil
import subprocess
import sys
import time
import torch
from road_training.common import ROOT, read, write, sha
from road_training.experiments.rcd_study.study import OUT, CORPUS, check_plan


def main():
    plan=check_plan();completed=[]
    def run(name,command):
        write(OUT/'progress.json',dict(state='running',active=name,completed=completed))
        started=time.monotonic()
        with (OUT/f'{name}.log').open('ab',buffering=0) as log:
            result=subprocess.run([sys.executable,'-m',*command],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write(OUT/'progress.json',dict(state='failed',active=name,completed=completed,returncode=result.returncode))
            raise RuntimeError(f'{name} failed; inspect its log')
        completed.append(dict(job=name,seconds=time.monotonic()-started))
    for seed in plan['seeds']:
        name=f'rcd_pretrain_seed{seed}';folder=OUT/name
        if not (folder/'progress.json').exists():
            recipe=plan['pretraining']
            run(name,['road_training.pretrain_rcd','--data-root',str(CORPUS),'--output',str(folder),
                '--seed',str(seed),'--epochs',str(recipe['epochs']),'--steps-per-epoch',str(recipe['steps_per_epoch']),
                '--batch-size',str(recipe['batch_size']),'--lr',str(recipe['lr']),
                '--weight-decay',str(recipe['weight_decay']),'--patience',str(recipe['patience'])])
        else:
            if read(folder/'progress.json')['state']!='complete':raise ValueError('Partial run requires explicit recovery')
            completed.append(dict(job=name,reused_completed=True))
        shutil.copy2(folder/'best.pt',folder/'selected.pt')
        selected=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
        write(folder/'selection.json',dict(epoch=selected['epoch'],sha256=sha(folder/'selected.pt'),
            rule=plan['pretraining']['selection']))
        for arm in plan['arms']:
            name=f'{arm}_seed{seed}';progress=OUT/name/'progress.json'
            if progress.exists():
                if read(progress)['state']!='complete':raise ValueError('Partial fine-tuning run requires explicit recovery')
                completed.append(dict(job=name,reused_completed=True));continue
            run(name,['road_training.experiments.rcd_study.fine_tune','--arm',arm,'--seed',str(seed)])
    write(OUT/'training_complete.json',dict(passed=True,jobs=completed,plan_sha256=sha(OUT/'plan.json')))
    for action in ['val','freeze','test']:
        run('evaluate_'+action,['road_training.experiments.rcd_study.evaluate',action])
    run('report',['road_training.experiments.rcd_study.report'])
    write(OUT/'progress.json',dict(state='complete',completed=completed,plan_sha256=sha(OUT/'plan.json')))


if __name__=='__main__':main()
