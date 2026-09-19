"""Train Time-RCD on labeled synthetic road recordings, without real labels.

Run: python -m road_training.pretrain_rcd --output PATH
The existing RoadDataset supplies raw signals, availability and sample labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader, RandomSampler
from road_training.dataset import RoadDataset
from road_training.rcd import RCDEncoder, RCDPretrainer


def update_patience(loss, significant_best, stale, minimum_improvement=1e-4):
    """Measure cumulative improvement since the last patience reset.

    Keep this reference separate from the minimum loss used to save weights.
    Otherwise several small improvements can be mistaken for a plateau.
    """
    if loss < significant_best - minimum_improvement:
        return loss, 0
    return significant_best, stale + 1


def run_epoch(model, loader, generator, optimizer=None, clean=False):
    training = optimizer is not None
    model.train(training)
    ce_sum = mse_sum = 0.
    labels_count = values_count = updates = tp = fp = fn = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            x, observed = batch['x'].cuda(non_blocking=True), batch['mask'].cuda(non_blocking=True)
            labels = batch['labels']['localized_disturbance'].cuda(non_blocking=True)
            patch_mask = (torch.zeros(x.shape[0],x.shape[1]//model.encoder.patch_length,
                          dtype=torch.bool,device=x.device) if clean else None)
            if training: optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(x, observed, labels, generator=generator, patch_mask=patch_mask)
            if not torch.isfinite(result['loss']): raise FloatingPointError('Nonfinite RCD objective')
            nc, nm = int(result['label_valid'].sum()), int(result['loss_mask'].sum())
            if training:
                if not nc or not nm: raise ValueError('RCD training requires anomaly and reconstruction targets')
                result['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();updates += 1
            with torch.no_grad():
                ce_sum += float(result['anomaly_loss'].detach())*nc
                mse_sum += float(result['reconstruction_loss'].detach())*nm
                labels_count += nc; values_count += nm
                valid = result['label_valid']
                truth, prediction = labels[valid].bool(), result['logits'][valid].argmax(-1).bool()
                tp += int((truth & prediction).sum());fp += int((~truth & prediction).sum());fn += int((truth & ~prediction).sum())
    if not labels_count: raise ValueError('No labeled anomaly targets')
    ce, mse = ce_sum/labels_count, mse_sum/max(values_count,1)
    return dict(loss=ce+mse,anomaly_ce=ce,masked_mse=mse,labeled_samples=labels_count,
        masked_values=values_count,optimizer_updates=updates,precision=tp/max(tp+fp,1),
        recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),clean_input=clean)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('road_training/data_transfer_v2_mount'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--steps-per-epoch',type=int,default=512)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--patience',type=int,default=7)
    args=p.parse_args()
    if args.output.exists():p.error('Preserve existing runs; choose a new output directory')
    if min(args.epochs,args.steps_per_epoch,args.batch_size,args.patience)<1 or args.workers<0 or args.lr<=0 or args.weight_decay<0:
        p.error('Invalid training settings')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():p.error('CUDA BF16 required')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    train=RoadDataset(args.data_root,source='synthetic',split='train',stride=1,return_labels=True)
    val=RoadDataset(args.data_root,source='synthetic',split='val',stride=1024,return_labels=True)
    # Use the same fixed normalizer as the preceding architecture controls.
    stats=RoadDataset(args.data_root,source='both',split='train',return_labels=False).train_stats
    model=RCDPretrainer(RCDEncoder(train_stats=stats)).cuda()
    sampler=RandomSampler(train,replacement=True,num_samples=args.steps_per_epoch*args.batch_size,
        generator=torch.Generator().manual_seed(args.seed+1))
    kwargs=dict(batch_size=args.batch_size,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    train_loader=DataLoader(train,sampler=sampler,generator=torch.Generator().manual_seed(args.seed+2),**kwargs)
    val_loader=DataLoader(val,shuffle=False,generator=torch.Generator().manual_seed(args.seed+3),**kwargs)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    config=dict(method='rcd',encoder_config=model.encoder.config,pretraining_config=model.config,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        manifest_sha256=hashlib.sha256((args.data_root/'manifest.json').read_bytes()).hexdigest(),
        train_statistics=stats,source='synthetic',normalization_source='both TRAIN',
        train_recordings=[r['path'] for r in train.records],val_recordings=[r['path'] for r in val.records],
        train_windows=len(train),val_windows=len(val),parameters=sum(p.numel() for p in model.parameters()),
        encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        selection='Minimum synthetic VAL anomaly CE + masked MSE; fixed VAL corruption; no real labels or TEST')
    args.output.mkdir(parents=True)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    rng=torch.Generator(device='cuda').manual_seed(args.seed+4)
    history=[];best=significant_best=float('inf');stale=0
    for epoch in range(1,args.epochs+1):
        started=time.monotonic()
        training=run_epoch(model,train_loader,rng,optimizer)
        if training['optimizer_updates']!=args.steps_per_epoch:raise RuntimeError('Incomplete epoch')
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            validation=run_epoch(model,val_loader,torch.Generator(device='cuda').manual_seed(args.seed+5))
            clean=run_epoch(model,val_loader,torch.Generator(device='cuda').manual_seed(args.seed+6),clean=True)
        row=dict(epoch=epoch,train=training,val=validation,clean_val=clean,seconds=time.monotonic()-started)
        history.append(row)
        saved=dict(method='rcd',encoder_config=model.encoder.config,pretraining_config=model.config,
            encoder_state=model.encoder.state_dict(),model_state=model.state_dict(),epoch=epoch,
            metrics=row,manifest_sha256=config['manifest_sha256'])
        torch.save(saved,args.output/'last.pt')
        significant_best,stale=update_patience(validation['loss'],significant_best,stale)
        if validation['loss']<best:
            best=validation['loss'];torch.save(saved,args.output/'best.pt')
        (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        (args.output/'progress.json').write_text(json.dumps(dict(state='running',epoch=epoch,best_loss=best,stale=stale))+'\n')
        print(f"RCD seed={args.seed} epoch={epoch} train={training['loss']:.4f} VAL CE={validation['anomaly_ce']:.4f} MSE={validation['masked_mse']:.4f} clean synthetic F1={clean['f1']:.4f}",flush=True)
        if stale>=args.patience:break
    (args.output/'progress.json').write_text(json.dumps(dict(state='complete',epochs=len(history),
        updates=sum(r['train']['optimizer_updates'] for r in history),best_loss=best,
        elapsed_seconds=sum(r['seconds'] for r in history),test_evaluated=False),indent=2)+'\n')


if __name__=='__main__':main()
