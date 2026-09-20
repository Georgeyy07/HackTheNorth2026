"""Controlled offline-ensemble to delayed-causal distillation experiment."""
import argparse
import gc
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from road_training.experiments.ensemble_teachers import OUT
from road_training.experiments.instance_study import DATA
from road_training.dataset import RoadDataset
from road_training.train_multitask import patch_targets, supervised_windows
from road_training.train import balanced_alpha
from road_training.experiments.real_mixup import dataset_pools, sample, save
from road_training.augmentation import perturb
from road_training.instance_loss import dataset_joint_loss
from road_training.streaming_model import StreamingRoadModel, aligned_output
from road_training.streaming_data import ContextWindows
from road_training.streaming_evaluation import evaluate, validation_loader, load_teachers, load_student, f1
from road_training.distillation_loss import distillation_loss
from road_training.common import read, write, sha

FOLDER = OUT/'students'


def initialize():
    FOLDER.mkdir(parents=True, exist_ok=True)
    if (FOLDER/'plan.json').exists():
        raise FileExistsError('Preserve the frozen student plan')
    paths = [Path(__file__), *[Path(__file__).resolve().parents[1] / n for n in (
        'streaming_model.py','streaming_data.py','streaming_evaluation.py','distillation_loss.py')]]
    paths += [Path(__file__).resolve().parents[1]/'tests/test_streaming.py']
    write(FOLDER/'plan.json', dict(arms={'supervised_d0':dict(delay=0,kd=0.),
        'distilled_d0':dict(delay=0,kd=.5), 'supervised_d2':dict(delay=2,kd=0.),
        'distilled_d2':dict(delay=2,kd=.5)},screen_seed=62,confirmation_seeds=[63,64],
        delay_selection='Highest mean seed-62 VAL F1 across supervised/distilled pair, then confirm BOTH methods at selected delay',
        model=dict(width=128,dilations=[1,2,4,8],normalization_history=16,dropout=.1),
        epochs=24,steps_per_epoch=256,batch_size=256,lr=1e-4,weight_decay=.01,
        samples_per_batch=dict(kaggle=64,lira=128,roadsens=64),
        context_patches=48,temperature=2.,distillation_warmup_epochs=4,
        minimum_epochs=6,patience=8,minimum_improvement=.001,
        normalization='Own stream rolling 16-patch observed moments; never global or future-window statistics',
        augmentation='Identical seed-specific rotation/noise on full context; teacher sees same augmented target crop',
        hard_loss='Dataset focal weights, Kaggle/RoadSens weights 1/.25, IRI Huber weight .25',
        soft_loss='Known task labels only, teacher mean softened probabilities, Bernoulli KL*T² + .25 IRI Huber',
        selection='Maximum VAL F1 at .5 threshold; lower VAL IRI breaks exact ties',
        refinement='Freeze encoder and detector; fit IRI head on clean LiRA TRAIN, epoch zero eligible, minimum VAL IRI',
        refinement_epochs=12,refinement_steps=128,
        context='Only same recording/split. Masked left padding; terminal patches needing nonexistent future are censored for all models.',
        latency='Full window and one-patch stateful CPU/GPU wall-clock p50/p95; delayed output measured at emission time',
        acceptance='Prefer causal if mean VAL F1 >= .70 and within .03 of ensemble, section IRI MAE <= 1.25*ensemble, compute p95 <160 ms; otherwise retain bidirectional',
        source_sha256={str(p.resolve()):sha(p) for p in paths},manifest_sha256=sha(DATA/'manifest.json')))


def check_plan():
    plan=read(FOLDER/'plan.json')
    assert sha(DATA/'manifest.json') == plan['manifest_sha256']
    for p,digest in plan['source_sha256'].items():
        assert sha(p)==digest, f'Frozen source changed: {p}'
    return plan


def training_data():
    base=RoadDataset(DATA,split='train',source='real',stride=1,return_labels=True,max_cached_recordings=128)
    selected,_,_=supervised_windows(base,16)
    pools=dataset_pools(base,selected.indices)
    alphas={}
    for name in ('kaggle','roadsens'):
        sub=RoadDataset(DATA,split='train',source='real',real_dataset=name,stride=1024,return_labels=True)
        _,counts,_=supervised_windows(sub,16)
        alphas[name]=balanced_alpha(counts).cuda()
    return base,ContextWindows(base),pools,alphas


def student_prediction(model,x,mask,batch):
    output=aligned_output(model(x,mask),48,model.delay_patches)
    output['patch_valid'] &= batch['context_available'][:,50:114].cuda()
    return output


def train(arm,seed,plan):
    folder=FOLDER/f'{arm}_seed{seed}'
    if (folder/'complete.json').exists():
        assert sha(folder/'best.pt')==read(folder/'complete.json')['checkpoint_sha256']
        return
    folder.mkdir(exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    setting=plan['arms'][arm]
    model=StreamingRoadModel(**plan['model'],delay_patches=setting['delay']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
    teacher=None
    if setting['kd']:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            teacher=load_teachers(OUT/'teacher_checkpoints.json')
        teacher.requires_grad_(False)
    base,wrapped,pools,alphas=training_data()
    val_data,val_loader=validation_loader('val')
    config=dict(arm=arm,seed=seed,setting=setting,model_config=model.config,
                plan_sha256=sha(FOLDER/'plan.json'),teacher_receipt_sha256=sha(OUT/'teacher_checkpoints.json'),
                training_alphas={k:v.tolist() for k,v in alphas.items()},parameters=sum(p.numel() for p in model.parameters()))
    write(folder/'config.json',config)
    history=[];best=(-1.,-float('inf'));significant=-1.;stale=0
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model_state']);optimizer.load_state_dict(saved['optimizer_state'])
        history,best,significant,stale=(saved[k] for k in ('history','best','significant','stale'))
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    start=time.monotonic()
    for epoch in range(len(history)+1,plan['epochs']+1):
        if epoch>plan['minimum_epochs'] and stale>=plan['patience']:break
        indices=sample(pools,plan['samples_per_batch'],seed,epoch,plan['steps_per_epoch'])
        loader=DataLoader(Subset(wrapped,indices),batch_size=256,num_workers=2,pin_memory=True,
                          generator=torch.Generator().manual_seed(seed*1000+epoch))
        model.train();totals=np.zeros(3)
        for step,batch in enumerate(loader):
            target={k:v.cuda(non_blocking=True) for k,v in patch_targets(batch,16).items()}
            x,mask=batch['context_x'].cuda(non_blocking=True),batch['context_mask'].cuda(non_blocking=True)
            x=perturb(x,mask,seed=seed,step=(epoch-1)*plan['steps_per_epoch']+step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=student_prediction(model,x,mask,batch)
                soft=output['roughness'].new_zeros(())
                if teacher is not None:
                    with torch.no_grad():
                        reference=teacher(x[:,768:1792],mask[:,768:1792],temperature=plan['temperature'])
                    soft=distillation_loss(output,reference,target,batch['dataset'],plan['temperature'])['total']
            hard=dataset_joint_loss(output,target,batch['dataset'],alphas,
                roughness_weight=.25,dataset_weights={'kaggle':1.,'roadsens':.25})['total']
            weight=setting['kd']*min(1.,epoch/plan['distillation_warmup_epochs'])
            loss=hard+weight*soft
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite student objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            totals += [float(loss.detach()),float(hard.detach()),float(soft.detach())]
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            metrics=evaluate(model,loader=val_loader,data=val_data,detailed=False)
        score=(f1(metrics),-metrics['roughness']['mae'])
        improved=score>best;best=max(best,score)
        if score[0]>significant+plan['minimum_improvement']:significant=score[0];stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,val=metrics,train_loss=(totals/len(loader)).tolist(),elapsed_seconds=time.monotonic()-start))
        state=dict(config=config,model_config=model.config,model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
            history=history,best=best,significant=significant,stale=stale,epoch=epoch,val_metrics=metrics,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
        save(folder/'last.pt',state)
        if improved:save(folder/'best.pt',state)
        write(folder/'history.json',history)
        write(OUT/'student_progress.json',dict(state='training',arm=arm,seed=seed,epoch=epoch,best_f1=best[0],pid=os.getpid()))
        print(f'{arm} seed={seed} epoch={epoch} VAL F1={score[0]:.4f} IRI={metrics["roughness"]["mae"]:.4f}',flush=True)
    write(folder/'complete.json',dict(checkpoint_sha256=sha(folder/'best.pt'),epochs=len(history),best_f1=best[0],
                                    elapsed_seconds=time.monotonic()-start))
    del model,teacher,optimizer,loader,val_loader,base,wrapped;gc.collect();torch.cuda.empty_cache()


def refine(arm,seed,plan):
    parent=FOLDER/f'{arm}_seed{seed}/best.pt';folder=FOLDER/f'{arm}_seed{seed}/roughness'
    if (folder/'complete.json').exists():return
    folder.mkdir(exist_ok=True)
    torch.manual_seed(seed);torch.set_num_threads(4)
    saved=torch.load(parent,map_location='cpu',weights_only=False)
    model=load_student(parent)
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('roughness_head.'))
    frozen={k:v.clone() for k,v in saved['model_state'].items() if not k.startswith('roughness_head.')}
    base=RoadDataset(DATA,source='real',real_dataset='lira',split='train',stride=1,return_labels=True)
    selected,_,_=supervised_windows(base,16);wrapped=ContextWindows(base)
    vd,vl=validation_loader('val')
    original=evaluate(model,loader=vl,data=vd,detailed=False)
    original_f1=f1(original);best=original['roughness']['mae'];stale=0
    save(folder/'best.pt',dict(saved,adaptation_epoch=0))
    optimizer=torch.optim.AdamW(model.roughness_head.parameters(),lr=1e-4,weight_decay=.01)
    rng=np.random.default_rng([seed,917,122]);history=[dict(epoch=0,val=original)]
    for epoch in range(1,plan['refinement_epochs']+1):
        if epoch>4 and stale>=4:break
        indices=rng.choice(selected.indices,plan['refinement_steps']*256,replace=True).tolist()
        loader=DataLoader(Subset(wrapped,indices),batch_size=256,num_workers=2,pin_memory=True,
                          generator=torch.Generator().manual_seed(seed+epoch))
        model.eval();model.roughness_head.train()
        for batch in loader:
            target={k:v.cuda() for k,v in patch_targets(batch,16).items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=student_prediction(model,batch['context_x'].cuda(),batch['context_mask'].cuda(),batch)
            valid=target['roughness_valid']&output['patch_valid']
            if not valid.any():continue
            loss=F.smooth_l1_loss(output['roughness'][valid],target['roughness'][valid],beta=1.)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.roughness_head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        metrics=evaluate(model,loader=vl,data=vd,detailed=False)
        assert f1(metrics)==original_f1
        error=metrics['roughness']['mae'];stale=0 if error<best-1e-4 else stale+1
        if error<best:
            best=error
            save(folder/'best.pt',dict(config=saved['config'],model_config=model.config,model_state=model.state_dict(),
                epoch=saved['epoch'],adaptation_epoch=epoch,val_metrics=metrics))
        history.append(dict(epoch=epoch,val=metrics));write(folder/'history.json',history)
        print(f'IRI {arm} seed={seed} epoch={epoch} MAE={error:.4f} fixed F1={original_f1:.4f}',flush=True)
    adapted=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(adapted['model_state'][k],v) for k,v in frozen.items())
    write(folder/'complete.json',dict(checkpoint_sha256=sha(folder/'best.pt'),frozen_tensors_verified=len(frozen),
        original_f1=original_f1,best_iri=best,parent_sha256=sha(parent)))
    del model,optimizer,loader,vl;gc.collect();torch.cuda.empty_cache()


def run():
    plan=check_plan()
    assert read(OUT/'teacher_progress.json')['state']=='complete'
    torch.set_num_threads(4)
    teacher=load_teachers(OUT/'teacher_checkpoints.json')
    write(OUT/'ensemble_val.json',evaluate(teacher,'val'))
    del teacher;gc.collect();torch.cuda.empty_cache()
    for arm in plan['arms']:
        train(arm,plan['screen_seed'],plan)
    pairs={delay:float(np.mean([read(FOLDER/f'{method}_d{delay}_seed62/complete.json')['best_f1']
                               for method in ('supervised','distilled')])) for delay in (0,2)}
    delay=max(pairs,key=pairs.get)
    write(FOLDER/'delay_decision.json',dict(delay_patches=delay,validation_pair_means=pairs,test_read=False))
    finalists=[f'{method}_d{delay}' for method in ('supervised','distilled')]
    for arm in finalists:
        for seed in plan['confirmation_seeds']:
            train(arm,seed,plan)
        for seed in [plan['screen_seed'],*plan['confirmation_seeds']]:
            refine(arm,seed,plan)
    checkpoints=[FOLDER/f'{arm}_seed{s}/roughness/best.pt' for arm in finalists for s in (62,63,64)]
    write(OUT/'evaluation_plan.json',dict(finalists=finalists,
        student_checkpoints={str(p):sha(p) for p in checkpoints},
        teacher_checkpoints=read(OUT/'teacher_checkpoints.json'),threshold=.5,test_read=False))
    for p in checkpoints:
        model=load_student(p)
        for split in ('val','test'):
            write(p.parent/f'{split}.json',evaluate(model,split))
        del model;gc.collect();torch.cuda.empty_cache()
    teacher=load_teachers(OUT/'teacher_checkpoints.json')
    write(OUT/'ensemble_test.json',evaluate(teacher,'test'))
    for i,member in enumerate(teacher.models):
        for split in ('val','test'):
            write(OUT/f'teacher_seed{52+i}_{split}.json',evaluate(member,split))
    write(OUT/'student_progress.json',dict(state='complete',finalists=finalists,pid=os.getpid()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.init:initialize()
    elif args.run:
        import fcntl
        with (FOLDER/'runner.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:run()
            except Exception:
                write(OUT/'student_failure.json',dict(error=traceback.format_exc()));raise
    else:parser.error('Choose --init or --run')
