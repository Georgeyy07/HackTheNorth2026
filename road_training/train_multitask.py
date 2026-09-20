"""Supervise roughness and localized disturbance at every PatchTST patch.

Example (repository root): python -m road_training.train_multitask --source both
Only TRAIN/VAL are loaded. Each head has its own validity mask and loss mean.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

if __package__:
    from road_training.dataset import RoadDataset
    from road_training.patchtst import PatchTST, PatchTSTRoadModel
    from road_training.train import balanced_alpha, classification_metrics, majority_patch_targets, focal_from_cross_entropy
else:
    from road_training.dataset import RoadDataset
    from road_training.patchtst import PatchTST, PatchTSTRoadModel
    from road_training.train import balanced_alpha, classification_metrics, majority_patch_targets, focal_from_cross_entropy

DISTURBANCE = "localized_disturbance"


def patch_targets(batch, patch_length):
    """Both targets are [B,N]; roughness must stay within one labeled section."""
    disturbance, disturbance_valid = majority_patch_targets(batch, DISTURBANCE, patch_length)
    labels = batch["labels"]
    b, t = labels["overall_iri"].shape
    if t % patch_length:
        raise ValueError("Time length must be divisible by patch_length")
    shape = (b, t // patch_length, patch_length)
    iri = labels["overall_iri"].reshape(shape)
    section = labels["roughness_section"].reshape(shape)
    valid = (labels["overall_iri_valid"] & batch["mask"].any(-1)).reshape(shape)
    if (valid & (~torch.isfinite(iri) | (iri < 0) | (section < 0))).any():
        raise ValueError("Valid roughness requires finite nonnegative IRI and a section ID")
    keep = valid.all(-1) & (section.amin(-1) == section.amax(-1))
    target = iri.masked_fill(~valid, 0).mean(-1).masked_fill(~keep, 0)
    return dict(roughness=target, roughness_valid=keep,
                disturbance=disturbance, disturbance_valid=disturbance_valid)


def joint_loss(output, targets, *, gamma=2., alpha=None, roughness_weight=1., disturbance_weight=1.):
    """Huber IRI loss + binary focal loss, each averaged on its own known labels.

    A missing task gets a constant zero, so its head has no gradient and AdamW
    does not decay it on a batch without its supervision.
    """
    prediction, logits = output["roughness"].float(), output["disturbance_logit"].float()
    if prediction.shape != targets["roughness"].shape or logits.shape != targets["disturbance"].shape:
        raise ValueError("Both heads and targets must have the same [batch,patch] shape")
    rv = targets["roughness_valid"] & output["patch_valid"]
    dv = targets["disturbance_valid"] & output["patch_valid"]
    rloss = dloss = prediction.new_zeros(())
    if rv.any():
        rloss = F.smooth_l1_loss(prediction[rv], targets["roughness"][rv], beta=1.)
    if dv.any():
        y = targets["disturbance"][dv]
        ce = F.binary_cross_entropy_with_logits(logits[dv], y.float(), reduction="none")
        loss = focal_from_cross_entropy(ce, gamma)
        if alpha is not None:
            loss = loss * alpha[y]
        dloss = loss.mean()
    return dict(total=roughness_weight*rloss + disturbance_weight*dloss,
                roughness=rloss, disturbance=dloss, roughness_valid=rv, disturbance_valid=dv)


def supervised_windows(dataset, patch_length):
    """Skip unlabeled windows using prefix sums, without reading sensor windows.

    Focal class weights use non-overlapping TRAIN patch counts, computed once
    per recording. They are not repeated according to sliding-window stride.
    """
    selected, counts, roughness_samples = [], np.zeros(2, np.int64), 0
    offset = 0
    for i, record in enumerate(dataset.records):
        arrays = dataset._open(i)
        observed = np.asarray(arrays["mask"]).any(1)
        y = np.asarray(arrays["labels"][:, 1])  # Canonical combined presence labels.
        dv = (y != -100) & observed
        rv = np.zeros(len(y), bool)
        if "overall_iri" in arrays:
            rv = np.isfinite(arrays["overall_iri"]) & (arrays["roughness_section"] >= 0) & observed
        starts = np.arange(0, record["samples"]-dataset.window_size+1, dataset.stride)
        prefix = np.r_[0, np.cumsum(dv | rv)]
        keep = prefix[starts+dataset.window_size] > prefix[starts]
        selected.extend((offset + np.flatnonzero(keep)).tolist())
        offset += len(starts)
        end = (int(starts[-1])+dataset.window_size)//patch_length*patch_length
        known = dv[:end].reshape(-1, patch_length)
        patches = y[:end].reshape(-1, patch_length)
        zero, one = [(known & (patches == c)).sum(1) for c in (0,1)]
        counts += [int((zero > one).sum()), int((one > zero).sum())]
        roughness_samples += int(rv[:end].sum())
    if not selected:
        raise ValueError(f"No supervised windows in {dataset.source}/{dataset.split}")
    return Subset(dataset, selected), counts.tolist(), roughness_samples


class EpochScores:
    """Accumulate each head on its own labeled patches, never average batch F1."""
    def __init__(self, device):
        self.nr = self.nd = 0
        self.rsum = self.dsum = self.absolute = self.squared = 0.
        self.confusion = torch.zeros(2,2,dtype=torch.long,device=device)

    def add(self, output, targets, losses):
        rv, dv = losses["roughness_valid"], losses["disturbance_valid"]
        nr, nd = int(rv.sum()), int(dv.sum())
        self.nr += nr; self.nd += nd
        self.rsum += losses["roughness"].item()*nr
        self.dsum += losses["disturbance"].item()*nd
        error = (output["roughness"][rv]-targets["roughness"][rv]).double()
        self.absolute += error.abs().sum().item(); self.squared += error.square().sum().item()
        prediction = (output["disturbance_logit"][dv] >= 0).long()
        pairs = targets["disturbance"][dv]*2+prediction
        self.confusion += torch.bincount(pairs,minlength=4).reshape(2,2)

    def result(self, roughness_weight, disturbance_weight):
        return dict(loss=roughness_weight*self.rsum/max(self.nr,1)+disturbance_weight*self.dsum/max(self.nd,1),
            metric_unit="patch", roughness=dict(labeled_patches=self.nr,
                loss=self.rsum/self.nr if self.nr else None, mae=self.absolute/self.nr if self.nr else None,
                rmse=(self.squared/self.nr)**.5 if self.nr else None, units="m/km"),
            disturbance=dict(labeled_patches=self.nd,loss=self.dsum/self.nd if self.nd else None,
                classification=classification_metrics(self.confusion,("no_disturbance","disturbance")) if self.nd else None))


def run_epoch(model, loader, device, optimizer=None, *, precision="bf16", gamma=2., alpha=None,
              roughness_weight=1., disturbance_weight=1., progress=None):
    device = torch.device(device)
    if precision not in ("fp32", "bf16") or (precision == "bf16" and device.type != "cuda"):
        raise ValueError("BF16 requires CUDA; use fp32 for CPU")
    training = optimizer is not None
    model.train(training)
    scores = EpochScores(device)
    by_source = {}
    by_dataset = {}
    loss_args = dict(gamma=gamma,alpha=alpha,roughness_weight=roughness_weight,disturbance_weight=disturbance_weight)
    with torch.set_grad_enabled(training):
        for step,batch in enumerate(loader,1):
            targets = {k:v.to(device,non_blocking=True) for k,v in patch_targets(batch,model.encoder.patch_length).items()}
            if not (targets["roughness_valid"].any() or targets["disturbance_valid"].any()):
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=precision=="bf16"):
                output = model(batch["x"].to(device,non_blocking=True),batch["mask"].to(device,non_blocking=True))
            losses = joint_loss(output,targets,**loss_args)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Non-finite multitask loss")
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
            with torch.no_grad():
                scores.add(output,targets,losses)
                for field,groups in [("source",by_source),("dataset",by_dataset)]:
                    for name in sorted(set(batch.get(field, []))):
                        select = torch.tensor([s==name for s in batch[field]],device=device)[:,None]
                        scoped = dict(targets,roughness_valid=targets["roughness_valid"] & select,
                                      disturbance_valid=targets["disturbance_valid"] & select)
                        groups.setdefault(name,EpochScores(device)).add(output,scoped,joint_loss(output,scoped,**loss_args))
            if progress is not None:
                progress(step,len(loader))
    if not scores.nr+scores.nd:
        raise ValueError("Epoch has no usable patch targets")
    metrics = scores.result(roughness_weight,disturbance_weight)
    metrics["by_source"] = {source:value.result(roughness_weight,disturbance_weight) for source,value in by_source.items()}
    metrics["by_dataset"] = {name:value.result(roughness_weight,disturbance_weight) for name,value in by_dataset.items()}
    return metrics


def write_metrics(writer, epoch, metrics, prefix=""):
    writer.add_scalar(prefix+"loss/total",metrics["loss"],epoch)
    for key in ["loss","mae","rmse"]:
        if metrics["roughness"][key] is not None:
            writer.add_scalar(f"{prefix}roughness/{key}",metrics["roughness"][key],epoch)
    d = metrics["disturbance"]
    if d["classification"] is not None:
        writer.add_scalar(prefix+"disturbance/loss",d["loss"],epoch)
        for key,value in d["classification"]["per_class"]["disturbance"].items():
            if key != "support":writer.add_scalar(f"{prefix}disturbance/{key}",value,epoch)
    for source,value in metrics.get("by_source",{}).items():
        write_metrics(writer,epoch,value,prefix+source+"/")
    for name,value in metrics.get("by_dataset",{}).items():
        write_metrics(writer,epoch,value,prefix+"dataset/"+name+"/")
    writer.flush()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root")
    p.add_argument("--source",choices=("real","synthetic","both"),default="both")
    p.add_argument("--val-source",choices=("real","synthetic","both"))
    p.add_argument("--real-dataset",choices=("all","kaggle","lira","roadsens"),default="all")
    p.add_argument("--val-real-dataset",choices=("all","kaggle","lira","roadsens"))
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--window-size",type=int,default=1024)
    p.add_argument("--stride",type=int,default=1)
    p.add_argument("--patch-length",type=int,default=16)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--weight-decay",type=float,default=.01)
    p.add_argument("--focal-gamma",type=float,default=2.)
    p.add_argument("--roughness-weight",type=float,default=1.)
    p.add_argument("--disturbance-weight",type=float,default=1.)
    p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--device",default="cuda")
    p.add_argument("--precision",choices=("bf16","fp32"),default="bf16")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--tensorboard",action="store_true")
    p.add_argument("--output")
    args=p.parse_args(argv)
    if min(args.epochs,args.batch_size,args.window_size,args.patch_length)<1 or args.window_size%args.patch_length:
        p.error("Positive sizes and window-size divisible by patch-length are required")
    if not 1<=args.stride<=args.window_size or args.num_workers<0:
        p.error("Invalid stride or num-workers")
    for name in ["lr","roughness_weight","disturbance_weight"]:
        if not np.isfinite(getattr(args,name)) or getattr(args,name)<=0:p.error(f"{name} must be finite and positive")
    if not np.isfinite(args.focal_gamma) or args.focal_gamma<0 or not np.isfinite(args.weight_decay) or args.weight_decay<0:
        p.error("focal-gamma and weight-decay must be finite and nonnegative")
    args.val_source=args.val_source or args.source
    args.val_real_dataset=args.val_real_dataset or args.real_dataset
    if args.precision=="bf16" and torch.device(args.device).type!="cuda":p.error("BF16 requires CUDA")
    return args


def epoch_progress(output, epoch, epochs, phase):
    """Write batch completion/ETA; loss and F1 remain completed-epoch metrics."""
    started=time.monotonic()

    def report(step, total):
        if step!=1 and step%100 and step!=total:
            return
        elapsed=time.monotonic()-started
        remaining=elapsed/step*(total-step)
        status=dict(state="running",epoch=epoch,epochs=epochs,phase=phase,
                    completed_batches=step,total_batches=total,elapsed_seconds=elapsed,
                    estimated_phase_seconds_remaining=remaining,
                    updated_at=datetime.now().astimezone().isoformat())
        path=output/"progress.pending.json"
        path.write_text(json.dumps(status,indent=2)+"\n")
        path.replace(output/"progress.json")
        if step==1 or step%500==0 or step==total:
            print(f"Epoch {epoch}/{epochs} {phase}: batch {step:,}/{total:,}, "
                  f"elapsed {elapsed/60:.1f} min, phase ETA {remaining/60:.1f} min",flush=True)
    return report


def main(argv=None):
    args=parse_args(argv)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    if device.type=="cuda":
        if not torch.cuda.is_available():raise ValueError("CUDA is unavailable")
        with torch.cuda.device(device):
            if args.precision=="bf16" and not torch.cuda.is_bf16_supported():raise ValueError("CUDA BF16 is unavailable")
    train=RoadDataset(args.data_root,source=args.source,split="train",window_size=args.window_size,stride=args.stride,return_labels=True,real_dataset=args.real_dataset)
    val=RoadDataset(args.data_root,source=args.val_source,split="val",window_size=args.window_size,stride=args.window_size,return_labels=True,real_dataset=args.val_real_dataset)
    training,counts,nrough=supervised_windows(train,args.patch_length)
    validation,_,val_rough=supervised_windows(val,args.patch_length)
    if not nrough:
        raise ValueError("No TRAIN overall roughness labels. Include prepared LiRA or synthetic data; Kaggle roughness is unknown.")
    if not val_rough:print("VAL has no measured overall roughness labels; only disturbance is evaluated and selects the checkpoint.",flush=True)
    alpha=(balanced_alpha(counts) if sum(counts) else torch.ones(2)).to(device)
    if not sum(counts):print("TRAIN has no disturbance labels; only the roughness head is supervised.",flush=True)
    encoder=PatchTST(patch_length=args.patch_length,max_patches=args.window_size//args.patch_length,train_stats=train.train_stats)
    model=PatchTSTRoadModel(encoder).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    loader_args=dict(batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=device.type=="cuda")
    train_loader=DataLoader(training,shuffle=True,generator=torch.Generator().manual_seed(args.seed),**loader_args)
    val_loader=DataLoader(validation,shuffle=False,**loader_args)
    config=dict(arguments=vars(args),encoder_config=encoder.config,train_statistics=train.train_stats,
        model="PatchTSTRoadModel",target_layout="batch,patch",disturbance_classes=["no_disturbance","disturbance"],
        roughness_target="LiRA: independently measured P79 section IRI; synthetic: geometry-derived overall section IRI including injected defects; units m/km",
        patch_policy=dict(disturbance="Union of all defect types, then majority; ties ignored",
                          roughness="Mean IRI in a fully labeled patch inside one section; boundary patches ignored"),
        loss=dict(roughness="SmoothL1 beta=1 m/km",disturbance="binary focal",gamma=args.focal_gamma,
                  alpha=alpha.cpu().tolist(),alpha_fit="Non-overlapping TRAIN patch counts",
                  roughness_weight=args.roughness_weight,disturbance_weight=args.disturbance_weight),
        selection_metric="val.loss",metric_unit="patch",train_windows=len(training),val_windows=len(validation),
        manifest_sha256=hashlib.sha256((train.root/"manifest.json").read_bytes()).hexdigest())
    output=Path(args.output) if args.output else Path(__file__).parent/"checkpoints"/f"multitask_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True,exist_ok=False)
    (output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    print(f"Two patch heads on {device}, {args.precision}: TRAIN {len(training):,} windows, VAL {len(validation):,}; saving {output.resolve()}",flush=True)
    history,best=[],float("inf")
    with ExitStack() as stack:
        writers={}
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            for split in ["train","val"]:
                writers[split]=SummaryWriter(str(output/"tensorboard"/split));stack.callback(writers[split].close)
        for epoch in range(1,args.epochs+1):
            start=time.monotonic()
            loss_args=dict(precision=args.precision,gamma=args.focal_gamma,alpha=alpha,
                           roughness_weight=args.roughness_weight,disturbance_weight=args.disturbance_weight)
            tr=run_epoch(model,train_loader,device,optimizer,
                         progress=epoch_progress(output,epoch,args.epochs,"train"),**loss_args)
            va=run_epoch(model,val_loader,device,
                         progress=epoch_progress(output,epoch,args.epochs,"val"),**loss_args)
            history.append(dict(epoch=epoch,seconds=time.monotonic()-start,train=tr,val=va))
            checkpoint=dict(epoch=epoch,config=config,model_state=model.state_dict(),val_metrics=va)
            torch.save(checkpoint,output/"last.pt")
            if va["loss"]<best:best=va["loss"];torch.save(checkpoint,output/"best.pt")
            (output/"history.json").write_text(json.dumps(history,indent=2,allow_nan=False)+"\n")
            for split,metrics in [("train",tr),("val",va)]:
                if split in writers:write_metrics(writers[split],epoch,metrics)
                d=metrics["disturbance"]["classification"]
                f1=d["per_class"]["disturbance"]["f1"] if d else None
                print(f"Epoch {epoch}/{args.epochs} {split}: loss={metrics['loss']:.4f}, roughness MAE={metrics['roughness']['mae']}, disturbance F1={f1}",flush=True)
                for source,part in metrics["by_source"].items():
                    d=part["disturbance"]["classification"]
                    f1=d["per_class"]["disturbance"]["f1"] if d else None
                    print(f"  {source}: roughness MAE={part['roughness']['mae']}, disturbance F1={f1}",flush=True)
                for name,part in metrics["by_dataset"].items():
                    d=part["disturbance"]["classification"]
                    f1=d["per_class"]["disturbance"]["f1"] if d else None
                    print(f"  dataset/{name}: roughness MAE={part['roughness']['mae']}, disturbance F1={f1}",flush=True)
    status=dict(state="complete",epoch=args.epochs,epochs=args.epochs,
                updated_at=datetime.now().astimezone().isoformat())
    (output/"progress.pending.json").write_text(json.dumps(status,indent=2)+"\n")
    (output/"progress.pending.json").replace(output/"progress.json")
    return output


if __name__=="__main__":main()
