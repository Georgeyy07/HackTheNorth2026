"""Wait for this suite's live process, then finish the VAL-only reports."""
import os
import subprocess
import sys
import time
from road_training.experiments.overfit.study import ROOT, OUT
from road_training.common import read, write


def main():
    pid=int((OUT/'suite.pid').read_text())
    while True:
        status=read(OUT/'progress.json') if (OUT/'progress.json').exists() else {'state':'starting'}
        if status['state']=='training_complete':break
        if status['state']=='failed':raise RuntimeError('Training failed; inspect the active run')
        try:os.kill(pid,0)
        except ProcessLookupError:raise RuntimeError('Training runner stopped before completion')
        time.sleep(15)
    for module in ['road_training.experiments.overfit.evaluate','road_training.experiments.overfit.report']:
        subprocess.run([sys.executable,'-m',module],cwd=ROOT,check=True)
    write(OUT/'finish_complete.json',dict(passed=True,test_read=False))


if __name__=='__main__':main()
