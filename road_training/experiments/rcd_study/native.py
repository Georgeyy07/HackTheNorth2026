"""Predeclared direct-transfer check of RCD's trained anomaly head (no IRI)."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from road_training.dataset import RoadDataset
from road_training.rcd import RCDEncoder, RCDPretrainer
from road_training.train import majority_patch_targets, classification_metrics
from road_training.common import ROOT, read, write, sha
from road_training.experiments.rcd_study.study import OUT, CORPUS, check_plan


def score(checkpoint, split):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=RCDPretrainer(RCDEncoder(**saved['encoder_config']),**saved['pretraining_config'])
    model.load_state_dict(saved['model_state']);model.cuda().eval()
    data=RoadDataset(CORPUS,source='real',real_dataset='kaggle',split=split,stride=1024,return_labels=True)
    loader=DataLoader(data,batch_size=256,num_workers=2,pin_memory=True)
    truth=[];scores=[];confusion=torch.zeros(2,2,dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            labels,keep=majority_patch_targets(batch,'localized_disturbance',16)
            x,observed=batch['x'].cuda(),batch['mask'].cuda()
            hidden=torch.zeros(x.shape[0],x.shape[1]//16,dtype=torch.bool,device=x.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=model(x,observed,patch_mask=hidden)
            probability=result['logits'].float().softmax(-1)[...,1].cpu().reshape(x.shape[0],-1,16)
            valid_time=batch['mask'].any(-1).reshape_as(probability)
            patch_probability=(probability*valid_time).sum(-1)/valid_time.sum(-1).clamp_min(1)
            y=labels[keep];p=patch_probability[keep]
            confusion+=torch.bincount(y*2+(p>=.5).long(),minlength=4).reshape(2,2)
            truth.extend(y.tolist());scores.extend(p.tolist())
    return dict(classification=classification_metrics(confusion,('normal','disturbance')),
        average_precision=float(average_precision_score(truth,scores)),split=split,
        checkpoint_sha256=sha(checkpoint),pretrain_epoch=saved['epoch'],threshold=.5,
        prediction='Mean native per-timestep anomaly probability over observed times in each 16-sample patch',
        target='Same majority-voted patch targets and non-overlapping windows as downstream evaluation',
        limitation='Real TRAIN-informed normalizer/simulator; synthetic-gradient direct transfer, not strict zero-shot; no roughness head')


def write_plan():
    path=OUT/'native_plan.json'
    if path.exists():raise FileExistsError('Preserve native evaluation plan')
    plan=check_plan()
    write(path,dict(created_utc=datetime.now(timezone.utc).isoformat(),seeds=plan['seeds'],
        selection='Use each pretraining selected.pt from minimum synthetic VAL CE+MSE; no real-label head fitting',
        threshold=.5,aggregation='Mean native per-sample anomaly probability over observed samples per16-sample patch',
        main_plan_sha256=sha(OUT/'plan.json'),native_source_sha256=sha(Path(__file__)),
        current_training_state=read(OUT/'progress.json'),test_evaluated=False))


def main():
    torch.set_num_threads(4)
    check_plan();plan=read(OUT/'native_plan.json')
    assert sha(Path(__file__))==plan['native_source_sha256']
    assert sha(OUT/'plan.json')==plan['main_plan_sha256']
    frozen_path=OUT/'native_frozen_checkpoints.json'
    if frozen_path.exists():raise FileExistsError('Do not repeat native TEST evaluation')
    checkpoints=[]
    for seed in plan['seeds']:
        folder=OUT/f'rcd_pretrain_seed{seed}'
        assert read(folder/'progress.json')['state']=='complete'
        checkpoints.append(dict(seed=seed,path=str(folder/'selected.pt'),sha256=sha(folder/'selected.pt')))
    write(frozen_path,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),checkpoints=checkpoints,
        plan_sha256=sha(OUT/'native_plan.json'),rule='All pretraining-selected native heads; no real VAL/TEST selection'))
    results={}
    for split in ['val','test']:
        rows=[]
        for checkpoint in checkpoints:
            assert sha(checkpoint['path'])==checkpoint['sha256']
            result=score(checkpoint['path'],split);rows.append(result)
            write(OUT/f"rcd_pretrain_seed{checkpoint['seed']}"/f'native_{split}.json',result)
        results[split]=dict(per_seed=rows)
        for key in ['precision','recall','f1']:
            values=[r['classification']['per_class']['disturbance'][key] for r in rows]
            results[split][key]=dict(mean=float(np.mean(values)),seed_sd=float(np.std(values,ddof=1)))
        values=[r['average_precision'] for r in rows]
        results[split]['average_precision']=dict(mean=float(np.mean(values)),seed_sd=float(np.std(values,ddof=1)))
    write(OUT/'native_comparison.json',results)
    print({split:{k:round(v['mean'],4) for k,v in r.items() if k!='per_seed'} for split,r in results.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',action='store_true')
    args=p.parse_args()
    if args.plan:write_plan()
    else:main()
