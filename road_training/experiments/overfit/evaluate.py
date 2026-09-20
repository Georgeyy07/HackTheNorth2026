"""Evaluate only completed, VAL-selected checkpoints on real/synthetic VAL."""
from road_training.experiments.overfit.study import OUT
from road_training.common import read, write, sha
from road_training.metrics import evaluate as events
from road_training.domain_evaluation import evaluate as patches


def main():
    plan=read(OUT/'plan.json');progress=read(OUT/'progress.json')
    if progress['state']!='training_complete':raise ValueError('Finish the declared suite first')
    rows=[]
    for arm in plan['arms']:
        for seed in plan['seeds']:
            folder=OUT/f'{arm}_seed{seed}';checkpoint=folder/'best.pt'
            events(checkpoint,folder/'event_section_val.json',split='val')
            if arm.startswith('M_'):
                patches(checkpoint,folder/'synthetic_val.json',source='synthetic',split='val')
            rows.append(dict(arm=arm,seed=seed,checkpoint_sha256=sha(checkpoint)))
            print('VAL evaluation',arm,seed,flush=True)
    write(OUT/'evaluation_complete.json',dict(passed=True,test_read=False,rows=rows,plan_sha256=sha(OUT/'plan.json')))


if __name__=='__main__':main()
