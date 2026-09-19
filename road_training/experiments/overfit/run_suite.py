"""Detached serial runner with progress receipts and no TEST evaluation."""
import subprocess
import sys
from road_training.experiments.overfit.study import OUT, ROOT
from road_training.common import read, write, sha


def main():
    plan=read(OUT/'plan.json');jobs=[(arm,seed) for seed in plan['seeds'] for arm in plan['arms']]
    done=[]
    for arm,seed in jobs:
        folder=OUT/f'{arm}_seed{seed}'
        if (folder/'progress.json').exists() and read(folder/'progress.json')['state']=='complete':
            done.append(dict(arm=arm,seed=seed));continue
        write(OUT/'progress.json',dict(state='running',active=dict(arm=arm,seed=seed),completed=done,total=len(jobs)))
        with (OUT/f'{arm}_seed{seed}.log').open('ab',buffering=0) as log:
            result=subprocess.run([sys.executable,'-m','road_training.experiments.overfit.study','--arm',arm,'--seed',str(seed)],
                                  cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write(OUT/'progress.json',dict(state='failed',active=dict(arm=arm,seed=seed),completed=done,returncode=result.returncode))
            raise RuntimeError(f'{arm} seed {seed} failed; inspect its log')
        done.append(dict(arm=arm,seed=seed))
    write(OUT/'progress.json',dict(state='training_complete',completed=done,total=len(jobs),plan_sha256=sha(OUT/'plan.json')))


if __name__=='__main__':main()
