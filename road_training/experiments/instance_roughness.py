"""Fit only the IRI head after selecting an instance-normalized detector.

All encoder/statistics/disturbance weights remain bit-identical. This plan
is declared before the initial study opens TEST. Selection uses VAL IRI only,
and the unmodified checkpoint is an eligible epoch-zero candidate.
"""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from road_training.experiments.instance_study import OUT as BASE, DATA, make_model, evaluate_checkpoint, f1, save
from road_training.experiments.instance_study import check_plan
from road_training.dataset import RoadDataset
from road_training.train_multitask import supervised_windows, patch_targets, run_epoch
from road_training.common import read, write, sha

OUT=BASE/'roughness_head_adaptation'


def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():raise FileExistsError('Preserve plan')
    if (BASE/'evaluation_plan.json').exists():raise RuntimeError('Declare head adaptation before initial TEST is opened')
    write(OUT/'plan.json',dict(trigger_mean_val_patch_mae=.45,epochs=12,steps_per_epoch=128,
        batch_size=256,lr=1e-4,weight_decay=.01,precision='bf16',seeds=[42,43,44],
        selection='Minimum full real-VAL patch IRI MAE; epoch-zero original is eligible',
        stopping='At least four epochs, then four checks without improvement of 1e-4',
        training='LiRA TRAIN only, instance normalization, frozen encoder/statistics/disturbance, train roughness_head only',
        evaluation='Freeze all adapted checkpoints before TEST; verify unchanged disturbance weights and VAL F1',
        base_plan_sha256=sha(BASE/'plan.json'),implementation_sha256=sha(__file__),
        test_read_at_declaration=False))


def train(seed,winner,plan):
    parent=BASE/f'{winner}_seed{seed}'/'best.pt';folder=OUT/f'{winner}_seed{seed}'
    if (folder/'complete.json').exists():return
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(seed)
    saved=torch.load(parent,map_location='cpu',weights_only=False);config=saved['config']
    model=make_model(config).cuda();model.load_state_dict(saved['model_state'])
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('roughness_head.'))
    frozen={k:v.clone() for k,v in saved['model_state'].items() if not k.startswith('roughness_head.')}
    optimizer=torch.optim.AdamW(model.roughness_head.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
    from road_training.acceleration_speed import AccelerationSpeedDataset
    dataset = AccelerationSpeedDataset if model.encoder.channels == 4 else RoadDataset
    training=dataset(DATA,source='real',real_dataset='lira',split='train',stride=1,return_labels=True)
    selected,_,_=supervised_windows(training,16)
    val=dataset(DATA,source='real',split='val',stride=1024,return_labels=True)
    validation,_,_=supervised_windows(val,16)
    loader_args=dict(batch_size=256,num_workers=2,pin_memory=True)
    val_loader=DataLoader(validation,shuffle=False,generator=torch.Generator().manual_seed(811),**loader_args)
    args=dict(precision='bf16',gamma=2.,alpha=torch.tensor(config['loss']['alpha'],device='cuda'))
    original=run_epoch(model,val_loader,'cuda',**args);original_f1=f1(original)
    assert original_f1==f1(saved['val_metrics'])
    best=original['roughness']['mae'];significant=best;stale=0;history=[dict(epoch=0,val=original)]
    # Keep the original task heads unless adaptation improves measured VAL IRI.
    save(folder/'best.pt',dict(saved,adaptation_epoch=0,parent_checkpoint_sha256=sha(parent)))
    rng=np.random.default_rng([seed,917,121]);start=time.monotonic()
    for epoch in range(1,plan['epochs']+1):
        if epoch>4 and stale>=4:break
        indices=rng.choice(selected.indices,plan['steps_per_epoch']*256,replace=True).tolist()
        loader=DataLoader(Subset(training,indices),shuffle=False,generator=torch.Generator().manual_seed(seed+epoch),**loader_args)
        model.eval();model.roughness_head.train()
        loss_sum=0.
        for batch in loader:
            target={k:v.cuda(non_blocking=True) for k,v in patch_targets(batch,16).items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(batch['x'].cuda(non_blocking=True),batch['mask'].cuda(non_blocking=True))
            valid=target['roughness_valid']&output['patch_valid']
            assert valid.any()
            loss=F.smooth_l1_loss(output['roughness'][valid],target['roughness'][valid],beta=1.)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.roughness_head.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();loss_sum+=float(loss.detach())
        metrics=run_epoch(model,val_loader,'cuda',**args)
        assert f1(metrics)==original_f1
        error=metrics['roughness']['mae']
        history.append(dict(epoch=epoch,val=metrics,train_iri_loss=loss_sum/len(loader)))
        if error<best:
            best=error
            save(folder/'best.pt',dict(config=config,model_state=model.state_dict(),epoch=saved['epoch'],
                adaptation_epoch=epoch,val_metrics=metrics,parent_checkpoint_sha256=sha(parent)))
        if error<significant-1e-4:significant=error;stale=0
        else:stale+=1
        write(folder/'history.json',history)
        print(f'IRI-head seed={seed} epoch={epoch} VAL IRI={error:.4f} fixed F1={original_f1:.4f}',flush=True)
    adapted=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(adapted['model_state'][k],v) for k,v in frozen.items())
    write(folder/'complete.json',dict(checkpoint_sha256=sha(folder/'best.pt'),parent_checkpoint_sha256=sha(parent),
        original_val_patch_iri_mae=original['roughness']['mae'],best_val_patch_iri_mae=best,
        unchanged_val_f1=original_f1,frozen_tensors_verified=len(frozen),best_epoch=adapted['adaptation_epoch'],
        elapsed_seconds=time.monotonic()-start))
    del model,optimizer,loader,val_loader;gc.collect();torch.cuda.empty_cache()


def run():
    check_plan()
    plan=read(OUT/'plan.json')
    assert sha(__file__)==plan['implementation_sha256']
    assert sha(BASE/'plan.json')==plan['base_plan_sha256']
    assert read(BASE/'progress.json')['state']=='complete'
    decision=read(BASE/'evaluation_plan.json');winner=decision['winner']
    original_errors=[]
    for seed in plan['seeds']:
        saved=torch.load(BASE/f'{winner}_seed{seed}'/'best.pt',map_location='cpu',weights_only=False)
        original_errors.append(saved['val_metrics']['roughness']['mae'])
    if np.mean(original_errors)<=plan['trigger_mean_val_patch_mae']:
        write(OUT/'progress.json',dict(state='not_needed',winner=winner,mean_val_mae=float(np.mean(original_errors))));return
    for seed in plan['seeds']:
        write(OUT/'progress.json',dict(state='training',seed=seed,winner=winner,pid=os.getpid()))
        train(seed,winner,plan)
    frozen={str(seed):sha(OUT/f'{winner}_seed{seed}'/'best.pt') for seed in plan['seeds']}
    write(OUT/'evaluation_plan.json',dict(winner=winner,checkpoints=frozen,selection='VAL IRI only; disturbance path unchanged'))
    for seed in plan['seeds']:
        for split in ('val','test'):
            folder=OUT/f'{winner}_seed{seed}'
            assert sha(folder/'best.pt')==frozen[str(seed)]
            write(OUT/'progress.json',dict(state='evaluating',seed=seed,split=split,pid=os.getpid()))
            evaluate_checkpoint(folder,split)
    write(OUT/'progress.json',dict(state='complete',winner=winner))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--init',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.init:initialize()
    elif args.run:
        try:run()
        except Exception:
            write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
    else:parser.error('Use --init or --run')
