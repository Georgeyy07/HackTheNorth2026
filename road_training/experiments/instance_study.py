"""Validation-led real-only study with per-instance normalization.

Run --init, then --run. Screen five declared recipes, confirm the two best
with two additional seeds, and evaluate the winning three frozen checkpoints.
"""
import argparse
from functools import partial
import gc
import os
from pathlib import Path
import random
import tarfile
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from road_training.dataset import RoadDataset
from road_training.instance_model import InstancePatchTST, InstanceRoadModel
from road_training.instance_loss import dataset_joint_loss
from road_training.augmentation import perturb
from road_training.experiments.real_mixup import dataset_pools, sample, save
from road_training.train import balanced_alpha
from road_training.train_multitask import patch_targets, supervised_windows, joint_loss, run_epoch, write_metrics, epoch_progress
from road_training.common import read, write, sha

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'road_training/data_with_roadsens'
OUT=ROOT/'reports/instance_roadsens_20260917'
ARMS={
    'instance_pooled': dict(statistics_mode='none', balance='pooled'),
    'instance_domains': dict(statistics_mode='none', balance='dataset'),
    'instance_roughness_stats': dict(statistics_mode='roughness', balance='dataset'),
    'instance_both_stats': dict(statistics_mode='both', balance='dataset'),
    'instance_detection_focus': dict(statistics_mode='both', balance='dataset',
        dataset_weights=dict(kaggle=1., roadsens=.25), roughness_weight=.25),
}


def initialize():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'plan.json').exists():raise FileExistsError('Preserve frozen plan')
    files=[Path(__file__),ROOT/'road_training/instance_model.py',ROOT/'road_training/instance_loss.py',
        ROOT/'road_training/patchtst.py',ROOT/'road_training/dataset.py',ROOT/'road_training/train_multitask.py',
        ROOT/'road_training/train.py',ROOT/'road_training/experiments/real_mixup.py',
        ROOT/'road_training/augmentation.py',ROOT/'road_training/metrics.py',
        ROOT/'road_training/tests/test_instance.py']
    plan=dict(arms=ARMS,screen_seed=42,confirmation_seeds=[43,44],finalists=2,
        epochs=24,steps_per_epoch=256,batch_size=256,lr=1e-4,weight_decay=.01,precision='bf16',device='cuda',
        samples_per_batch=dict(kaggle=64,lira=128,roadsens=64),
        source='real',window_size=1024,patch_length=16,instance_eps=1e-5,
        normalization='Per-window, per-channel observed mean/variance; no global statistics; no running state',
        statistics_branch='Optional same-window asinh(mean), log(std), observed fraction; never dataset fitted',
        augmentation='Shared <=5 degree IMU rotation and existing small held-sample noise/bias',
        loss='Huber IRI + focal gamma=2; per-dataset class weights from non-overlapping real TRAIN patches',
        selection='Maximum Kaggle VAL disturbance F1 at fixed 0.5 threshold; VAL loss breaks ties',
        early_stopping=dict(minimum_updates=1536,patience_epochs=8,minimum_f1_improvement=.001),
        final_selection='Top two seed-42 recipes confirmed on seeds 43/44; highest mean VAL F1 wins; TEST only winning recipe after all checkpoint hashes freeze',
        data_root=str(DATA),manifest_sha256=sha(DATA/'manifest.json'),
        source_sha256={str(p):sha(p) for p in files},
        limitations=['Kaggle VAL/TEST are purged chronological Larisa segments; LiRA roads M3/M13 are held out.',
                    'RoadSens is TRAIN-only; no unseen RoadSens or city/device generalization claim.',
                    'Repeated validation-guided model development can overfit the development road.',
                    'Window normalization is bidirectional within the observed input, not causal; no hidden-patch pretraining.',
                    'TEST has historical exposure; no thresholds or settings are selected on current TEST.'])
    write(OUT/'plan.json',plan)
    with tarfile.open(OUT/'sources.tar.gz','w:gz') as archive:
        for p in files:archive.add(p,arcname=str(p.relative_to(ROOT)))
    return plan


def check_plan():
    p=read(OUT/'plan.json')
    assert sha(DATA/'manifest.json')==p['manifest_sha256']
    for path,digest in p['source_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'Frozen implementation changed: {path}')
    return p


def f1(metrics):
    return metrics['by_dataset']['kaggle']['disturbance']['classification']['per_class']['disturbance']['f1']


def make_model(config):
    return InstanceRoadModel(InstancePatchTST(**config['encoder_config']),**config['model_config'])


def train_epoch(model,loader,optimizer,alphas,pooled,setting,seed,epoch,plan,progress):
    model.train(); sums=dict(total=0.,roughness=0.,disturbance=0.); nr=nd=0
    for step,batch in enumerate(loader):
        targets={k:v.cuda(non_blocking=True) for k,v in patch_targets(batch,16).items()}
        x,mask=batch['x'].cuda(non_blocking=True),batch['mask'].cuda(non_blocking=True)
        x=perturb(x,mask,seed=seed,step=(epoch-1)*plan['steps_per_epoch']+step,rotate=True,noise=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):output=model(x,mask)
        if setting['balance']=='pooled':
            loss=joint_loss(output,targets,alpha=pooled)
        else:
            loss=dataset_joint_loss(output,targets,batch['dataset'],alphas,
                dataset_weights=setting.get('dataset_weights'),roughness_weight=setting.get('roughness_weight',1.))
        if not torch.isfinite(loss['total']):raise FloatingPointError('Nonfinite training loss')
        loss['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        for name in sums:sums[name]+=float(loss[name].detach())
        nr+=int(loss['roughness_valid'].sum());nd+=int(loss['disturbance_valid'].sum())
        progress(step+1,len(loader))
    return dict(**{k:v/len(loader) for k,v in sums.items()},roughness_patches=nr,
                disturbance_patches=nd,optimizer_updates=len(loader))


def train(arm,seed):
    plan=check_plan(); setting=plan['arms'][arm]; folder=OUT/f'{arm}_seed{seed}'
    if (folder/'complete.json').exists():
        assert sha(folder/'best.pt')==read(folder/'complete.json')['checkpoint_sha256'];return
    folder.mkdir(exist_ok=True)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    data=RoadDataset(DATA,source='real',split='train',stride=1,return_labels=True,max_cached_recordings=128)
    selected,counts,_=supervised_windows(data,16); pools=dataset_pools(data,selected.indices)
    assert all(len(p) for p in pools.values())
    per_dataset={}
    for name in ('kaggle','roadsens'):
        sub=RoadDataset(DATA,source='real',real_dataset=name,split='train',stride=1024,return_labels=True)
        _,c,_=supervised_windows(sub,16);per_dataset[name]=c
    alphas={k:balanced_alpha(v).cuda() for k,v in per_dataset.items()};pooled=balanced_alpha(counts).cuda()
    encoder=InstancePatchTST(max_patches=64,instance_eps=plan['instance_eps'])
    model=InstanceRoadModel(encoder,statistics_mode=setting['statistics_mode']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
    config=dict(model='InstanceRoadModel',arm=arm,seed=seed,source='real',data_root=str(DATA),
        manifest_sha256=plan['manifest_sha256'],plan_sha256=sha(OUT/'plan.json'),setting=setting,
        encoder_config=encoder.config,model_config=dict(statistics_mode=setting['statistics_mode']),
        train_statistics=None,normalization=plan['normalization'],
        parameters=sum(p.numel() for p in model.parameters()),arguments=dict(window_size=1024,lr=plan['lr'],batch_size=256,precision='bf16'),
        loss=dict(gamma=2.,alpha=alphas['kaggle'].tolist(),roughness_weight=1.,disturbance_weight=1.),
        training_alphas={k:v.tolist() for k,v in alphas.items()},pooled_alpha=pooled.tolist(),train_patch_counts=per_dataset,
        windows_by_dataset={k:len(v) for k,v in pools.items()},samples_per_batch=plan['samples_per_batch'],
        train_recordings=[r['id'] for r in data.records])
    if (folder/'config.json').exists() and read(folder/'config.json')!=config:raise ValueError('Resume configuration changed')
    write(folder/'config.json',config)
    history=[];best=(-1.,-float('inf'));significant=-1.;stale=0;loader=None
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model_state']);optimizer.load_state_dict(saved['optimizer_state'])
        history,best,significant,stale=(saved[k] for k in ('history','best_score','significant','stale'))
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    val=RoadDataset(DATA,source='real',split='val',stride=1024,return_labels=True)
    validation,_,_=supervised_windows(val,16)
    kwargs=dict(batch_size=256,num_workers=2,pin_memory=True)
    val_loader=DataLoader(validation,shuffle=False,generator=torch.Generator().manual_seed(817),**kwargs)
    probe=DataLoader(Subset(data,sample(pools,plan['samples_per_batch'],817,0,4)),shuffle=False,
                     generator=torch.Generator().manual_seed(818),**kwargs)
    eval_args=dict(precision='bf16',gamma=2.,alpha=alphas['kaggle'])
    early=plan['early_stopping']; start=time.monotonic()
    with SummaryWriter(str(folder/'tensorboard')) as writer:
        for epoch in range(len(history)+1,plan['epochs']+1):
            if history and history[-1]['updates']>=early['minimum_updates'] and stale>=early['patience_epochs']:break
            indices=sample(pools,plan['samples_per_batch'],seed,epoch,plan['steps_per_epoch'])
            loader=DataLoader(Subset(data,indices),shuffle=False,generator=torch.Generator().manual_seed(seed*1000+epoch),**kwargs)
            objective=train_epoch(model,loader,optimizer,alphas,pooled,setting,seed,epoch,plan,
                epoch_progress(folder,epoch,plan['epochs'],'train'))
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                metrics=run_epoch(model,val_loader,'cuda',**eval_args)
                clean=run_epoch(model,probe,'cuda',**eval_args)
            score=(f1(metrics),-metrics['loss']); improved=score>best;best=max(best,score)
            if score[0]>significant+early['minimum_f1_improvement']:significant=score[0];stale=0
            else:stale+=1
            history.append(dict(epoch=epoch,updates=epoch*plan['steps_per_epoch'],objective=objective,val=metrics,
                train_clean=clean,elapsed_seconds=time.monotonic()-start))
            state=dict(config=config,model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),epoch=epoch,
                history=history,best_score=best,significant=significant,stale=stale,val_metrics=metrics,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
            save(folder/'last.pt',state)
            if improved:save(folder/'best.pt',state)
            write(folder/'history.json',history)
            write_metrics(writer,epoch,metrics,'val/');write_metrics(writer,epoch,clean,'train_clean/')
            writer.add_scalar('train_augmented/loss',objective['total'],epoch);writer.flush()
            write(folder/'progress.json',dict(state='training',epoch=epoch,val_f1=score[0],best_f1=best[0],stale=stale))
            print(f'{arm} seed={seed} epoch={epoch} VAL F1={score[0]:.4f} IRI={metrics["roughness"]["mae"]:.4f}',flush=True)
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    write(folder/'complete.json',dict(checkpoint_sha256=sha(folder/'best.pt'),epochs=len(history),
        best_epoch=saved['epoch'],best_f1=best[0],updates=history[-1]['updates'],elapsed_seconds=time.monotonic()-start))
    del loader,val_loader,probe,optimizer,model;gc.collect();torch.cuda.empty_cache()


def evaluate_checkpoint(folder,split):
    from road_training.metrics import evaluate as detailed
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);c=saved['config']
    model=make_model(c).cuda();model.load_state_dict(saved['model_state'])
    data=RoadDataset(DATA,source='real',split=split,stride=1024,return_labels=True)
    selected,_,_=supervised_windows(data,16)
    loader=DataLoader(selected,batch_size=256,num_workers=2,pin_memory=True,shuffle=False)
    metrics=run_epoch(model,loader,'cuda',precision='bf16',gamma=2.,alpha=torch.tensor(c['loss']['alpha'],device='cuda'))
    write(folder/f'{split}_patches.json',dict(metrics=metrics,epoch=saved['epoch'],split=split,
        checkpoint_sha256=sha(folder/'best.pt'),normalization=c['normalization'],threshold=.5,windows=len(selected)))
    del model,loader;gc.collect();torch.cuda.empty_cache()
    detailed(folder/'best.pt',folder/f'{split}_details.json',split=split,encoder_class=InstancePatchTST,
             model_class=partial(InstanceRoadModel,**c['model_config']))


def run():
    plan=check_plan()
    for arm in plan['arms']:
        write(OUT/'progress.json',dict(state='screening',arm=arm,seed=42,pid=os.getpid()))
        train(arm,42)
    ranked=sorted(plan['arms'],key=lambda arm:read(OUT/f'{arm}_seed42/complete.json')['best_f1'],reverse=True)
    finalists=ranked[:plan['finalists']]
    decision=dict(ranked_screen=ranked,finalists=finalists,criterion=plan['selection'],test_read=False)
    if (OUT/'screen_decision.json').exists():assert read(OUT/'screen_decision.json')==decision
    write(OUT/'screen_decision.json',decision)
    for arm in finalists:
        for seed in plan['confirmation_seeds']:
            write(OUT/'progress.json',dict(state='confirming',arm=arm,seed=seed,pid=os.getpid()))
            train(arm,seed)
    means={arm:float(np.mean([read(OUT/f'{arm}_seed{seed}/complete.json')['best_f1'] for seed in (42,43,44)])) for arm in finalists}
    winner=max(means,key=means.get)
    frozen=dict(winner=winner,validation_means=means,threshold=.5,
        checkpoints={str(seed):sha(OUT/f'{winner}_seed{seed}/best.pt') for seed in (42,43,44)},plan_sha256=sha(OUT/'plan.json'))
    if (OUT/'evaluation_plan.json').exists():assert read(OUT/'evaluation_plan.json')==frozen
    write(OUT/'evaluation_plan.json',frozen)
    for seed in (42,43,44):
        folder=OUT/f'{winner}_seed{seed}'
        for split in ('val','test'):
            check_plan();assert sha(folder/'best.pt')==frozen['checkpoints'][str(seed)]
            write(OUT/'progress.json',dict(state='evaluating',arm=winner,seed=seed,split=split,pid=os.getpid()))
            if not (folder/f'{split}_details.json').exists():evaluate_checkpoint(folder,split)
    write(OUT/'progress.json',dict(state='complete',winner=winner,validation_means=means,pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--init',action='store_true');p.add_argument('--run',action='store_true')
    args=p.parse_args()
    if args.init:initialize()
    elif args.run:
        import fcntl
        with (OUT/'runner.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:run()
            except Exception:
                write(OUT/'failure.json',dict(error=traceback.format_exc(),pid=os.getpid()))
                write(OUT/'progress.json',dict(state='failed',pid=os.getpid()));raise
    else:p.error('Use --init or --run')
