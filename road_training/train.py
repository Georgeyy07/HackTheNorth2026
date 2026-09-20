"""Train PatchTST from scratch on majority-voted patch classification labels.

Example: python train.py --source real --target kaggle_type --epochs 20
Only TRAIN and VAL are loaded. See README.md for target definitions.
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
    from road_training.dataset import KAGGLE_CLASSES, RoadDataset
    from road_training.patchtst import PatchTST, PatchTSTPredictor
else:
    from road_training.dataset import KAGGLE_CLASSES, RoadDataset
    from road_training.patchtst import PatchTST, PatchTSTPredictor


CLASSES = {
    "localized_disturbance": ("no_disturbance", "disturbance"),
    "defect": ("normal", "defect"),
    "defect_assumed_normal": ("normal", "defect"),
    "kaggle_type": KAGGLE_CLASSES,
    "synthetic_type": ("pothole", "speed_bump", "crack"),
    "quality_grade": ("good", "medium", "bad", "terrible"),
}


def balanced_alpha(counts):
    """Inverse TRAIN frequency, with mean weight 1 over training patch targets.

    An absent training class gets neutral weight 1, so it is not silently
    excluded if it appears in validation. Counts include window overlap.
    """
    counts = torch.as_tensor(counts, dtype=torch.float32)
    if not torch.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("Class counts must be finite, nonnegative and nonempty")
    present = counts > 0
    alpha = torch.ones_like(counts)
    alpha[present] = counts.sum() / (present.sum() * counts[present])
    return alpha


def focal_from_cross_entropy(ce, gamma):
    """Apply focal modulation without singular gradients at rounded-zero CE."""
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("Focal gamma must be finite and nonnegative")
    factor = -torch.expm1(-ce)
    if 0 < gamma < 1:
        # x**gamma has an infinite derivative at zero for fractional gamma.
        # Confident predictions (and ignored labels) can have exactly zero CE.
        # The smallest normal value also protects subnormal CE gradients.
        factor = factor.clamp_min(torch.finfo(ce.dtype).tiny)
    return factor.pow(gamma) * ce


def focal_loss(logits, labels, *, gamma=2.0, alpha=None):
    """Softmax focal loss with classes last; returns the shape of labels.

    The trainer uses patch logits [B,N,C] with labels [B,N]. Arbitrary leading
    dimensions, including flat [M,C] logits with labels [M], are supported.

    alpha[y] * (1 - p[y])**gamma * -log(p[y]); unknown (-100) is ignored.
    Compute p[y] from UNWEIGHTED cross-entropy, then apply class weights.
    """
    if labels.ndim < 1 or logits.shape[:-1] != labels.shape:
        raise ValueError("Logits must have shape [*labels.shape, classes]")
    ce = F.cross_entropy(logits.movedim(-1, 1), labels, reduction="none", ignore_index=-100)
    loss = focal_from_cross_entropy(ce, gamma)
    if alpha is not None:
        loss = loss * alpha[labels.clamp(min=0)]
    return loss


def target_mask(batch, target):
    """Unknown labels and times with no observed inputs cannot contribute."""
    return (batch["labels"][target] != -100) & batch["mask"].any(-1)


def group_patch_targets(batch, target, patch_length):
    """[B,T] labels/mask -> [B,T/P,P], aligned with the input patches.

    Only reshape: preserve every class ID, -100, and within-patch transition.
    """
    labels = batch["labels"][target]
    if labels.ndim != 2 or type(patch_length) is not int or patch_length < 1:
        raise ValueError("Expected labels [batch,time] and a positive integer patch_length")
    b, t = labels.shape
    if not t or t % patch_length:
        raise ValueError("Label length must be nonempty and divisible by patch_length")
    shape = (b, t // patch_length, patch_length)
    return labels.reshape(shape), target_mask(batch, target).reshape(shape)


def majority_patch_targets(batch, target, patch_length):
    """[B,T] -> one class ID and validity flag per patch, both [B,T/P].

    Vote only with known labels at times with observed inputs. Use the unique
    most frequent class (normal participates); ties and empty votes are -100.
    One valid vote is sufficient; no minimum coverage threshold is imposed.
    """
    labels, observed = group_patch_targets(batch, target, patch_length)
    classes = len(CLASSES[target])
    if (((labels < 0) | (labels >= classes)) & observed).any():
        raise ValueError(f"Invalid class ID for {target}")
    votes = torch.stack([((labels == c) & observed).sum(-1) for c in range(classes)], dim=-1)
    most_votes, majority = votes.max(-1)
    keep = (most_votes > 0) & ((votes == most_votes.unsqueeze(-1)).sum(-1) == 1)
    return majority.masked_fill(~keep, -100), keep


def labeled_windows(dataset, target, patch_length):
    """Keep windows with voted targets; count PATCH labels for focal weights."""
    indices = []
    counts = torch.zeros(len(CLASSES[target]), dtype=torch.long)
    for index in range(len(dataset)):
        batch = dataset[index]
        if not target_mask(batch, target).any():
            continue
        batched = {"mask": batch["mask"].unsqueeze(0),
                   "labels": {target: batch["labels"][target].unsqueeze(0)}}
        labels, keep = majority_patch_targets(batched, target, patch_length)
        labels = labels[keep]
        if labels.numel():
            indices.append(index)
            counts += torch.bincount(labels, minlength=len(counts))
    if not indices:
        raise ValueError(f"No usable {target} labels in {dataset.source}/{dataset.split}")
    return Subset(dataset, indices), counts.tolist()


def classification_metrics(confusion, classes):
    """Rows=true, columns=predicted. Undefined ratios are reported as zero.

    Macro F1 includes the fixed task taxonomy, even classes absent in VAL.
    Compute once from the full epoch's counts, never by averaging batch F1s.
    """
    counts = confusion.cpu().long()
    matrix = counts.double()
    correct = matrix.diag()
    support, predicted = matrix.sum(1), matrix.sum(0)
    precision = correct / predicted.clamp(min=1)
    recall = correct / support.clamp(min=1)
    f1 = 2 * correct / (support + predicted).clamp(min=1)
    metrics = {
        "accuracy": (correct.sum() / matrix.sum().clamp(min=1)).item(),
        "macro_f1": f1.mean().item(),
        "labeled_patches": counts.sum().item(),
        "metric_unit": "patch",
        "per_class": {
            name: {"precision": precision[i].item(), "recall": recall[i].item(),
                   "f1": f1[i].item(), "support": int(support[i].item())}
            for i, name in enumerate(classes)
        },
        "confusion_matrix": counts.tolist(),
    }
    if tuple(classes) == KAGGLE_CLASSES:
        # Any predicted defect counts as detection, even if its type is wrong.
        binary = torch.stack((counts[0, 0], counts[0, 1:].sum(),
                              counts[1:, 0].sum(), counts[1:, 1:].sum())).reshape(2, 2)
        metrics["binary_defect"] = classification_metrics(binary, ("normal", "defect"))
    return metrics


def run_epoch(model, loader, target, device, optimizer=None, *, gamma=2.0, alpha=None,
              precision="fp32"):
    """An optimizer enables training; BF16 autocast leaves weights/loss in FP32."""
    device = torch.device(device)
    if precision not in ("bf16", "fp32") or (precision == "bf16" and device.type != "cuda"):
        raise ValueError("Use precision='bf16' on CUDA, or precision='fp32'")
    training = optimizer is not None
    model.train(training)
    classes = CLASSES[target]
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.long, device=device)
    total_loss, total_patches = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            labels, keep = majority_patch_targets(batch, target, model.encoder.patch_length)
            labels, keep = labels.to(device, non_blocking=True), keep.to(device, non_blocking=True)
            count = int(keep.sum().item())
            if not count:
                continue
            x = batch["x"].to(device, non_blocking=True)
            observed = batch["mask"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                logits = model(x, observed)  # [B, patches, classes]
            # Keep softmax/focal arithmetic and reduction in FP32, outside autocast.
            loss_sum = focal_loss(logits.float(), labels, gamma=gamma, alpha=alpha).sum()
            if not torch.isfinite(loss_sum):
                raise FloatingPointError("Non-finite loss; check inputs and learning rate")
            if training:
                (loss_sum / count).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            total_loss += loss_sum.detach().item()
            total_patches += count
            pairs = labels[keep] * len(classes) + logits.detach().argmax(-1)[keep]
            confusion += torch.bincount(pairs, minlength=len(classes)**2).reshape_as(confusion)
    if not total_patches:
        raise ValueError("Epoch contains no usable targets")
    metrics = classification_metrics(confusion, classes)
    metrics["loss"] = total_loss / total_patches
    return metrics


def write_epoch_metrics(writer, epoch, metrics):
    """Write completed-epoch patch scores; TRAIN and VAL use separate writers."""
    writer.add_scalar("loss/focal", metrics["loss"], epoch)
    writer.add_scalar("f1/macro", metrics["macro_f1"], epoch)
    for name, scores in metrics["per_class"].items():
        writer.add_scalar(f"f1_per_class/{name}", scores["f1"], epoch)
    if "binary_defect" in metrics:
        defect_f1 = metrics["binary_defect"]["per_class"]["defect"]["f1"]
        writer.add_scalar("f1/binary_defect", defect_f1, epoch)
    elif "defect" in metrics["per_class"]:
        writer.add_scalar("f1/binary_defect", metrics["per_class"]["defect"]["f1"], epoch)
    elif "disturbance" in metrics["per_class"]:
        writer.add_scalar("f1/localized_disturbance", metrics["per_class"]["disturbance"]["f1"], epoch)
    writer.flush()  # Make this epoch visible while the next one is training.


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="Defaults to this directory's data folder")
    parser.add_argument("--source", choices=("real", "synthetic", "both"), default="real")
    parser.add_argument("--val-source", choices=("real", "synthetic", "both"),
                        help="Defaults to --source; always uses the training normalizer")
    parser.add_argument("--real-dataset", choices=("all", "kaggle", "lira", "roadsens"), default="all")
    parser.add_argument("--val-real-dataset", choices=("all", "kaggle", "lira", "roadsens"))
    parser.add_argument("--target", choices=tuple(CLASSES), default="kaggle_type")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1, help="TRAIN stride in samples; default: 1 (10 ms)")
    parser.add_argument("--patch-length", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", choices=("balanced", "none"), default="balanced",
                        help="Class weights from TRAIN patch counts (default) or unweighted focal loss")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", help="Default: cuda; CPU requires --precision fp32")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16",
                        help="Default: CUDA bfloat16 mixed precision; fp32 disables autocast")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard", action="store_true",
                        help="Save live TRAIN/VAL curves after each epoch (requires tensorboard)")
    parser.add_argument("--output", help="New run directory; default: checkpoints/<target>_<timestamp>")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.window_size, args.patch_length) < 1:
        parser.error("epochs, batch-size, window-size and patch-length must be positive")
    if args.window_size % args.patch_length:
        parser.error("window-size must be divisible by patch-length")
    if not 1 <= args.stride <= args.window_size:
        parser.error("stride must be between 1 and window-size")
    if args.num_workers < 0 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("num-workers must be nonnegative and lr must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    if not np.isfinite(args.focal_gamma) or args.focal_gamma < 0:
        parser.error("focal-gamma must be finite and nonnegative")
    args.val_source = args.val_source or args.source
    args.val_real_dataset = args.val_real_dataset or args.real_dataset
    if args.precision == "bf16" and torch.device(args.device).type != "cuda":
        parser.error("bf16 training requires CUDA; for CPU debugging use --device cpu --precision fp32")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise SystemExit("Live curves require TensorBoard: python -m pip install 'tensorboard>=2.14,<3'") from error
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; training defaults to GPU. For CPU debugging use --device cpu --precision fp32")
    if device.type == "cuda":
        with torch.cuda.device(device):
            if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
                raise ValueError("This CUDA device does not support bfloat16; use --precision fp32")
            print(f"GPU: {torch.cuda.get_device_name(device)}; precision: {args.precision}", flush=True)
    classes = CLASSES[args.target]
    train = RoadDataset(args.data_root, split="train", source=args.source,
                        window_size=args.window_size, stride=args.stride, return_labels=True, real_dataset=args.real_dataset)
    val = RoadDataset(args.data_root, split="val", source=args.val_source,
                      window_size=args.window_size, stride=args.window_size, return_labels=True, real_dataset=args.val_real_dataset)
    train_subset, train_counts = labeled_windows(train, args.target, args.patch_length)
    val_subset, val_counts = labeled_windows(val, args.target, args.patch_length)
    alpha = balanced_alpha(train_counts).to(device) if args.focal_alpha == "balanced" else None
    loss_args = dict(gamma=args.focal_gamma, alpha=alpha, precision=args.precision)
    loss_config = dict(name="focal", gamma=args.focal_gamma,
                       alpha=None if alpha is None else alpha.cpu().tolist(),
                       alpha_fit="TRAIN valid majority-patch target occurrences, including overlapping windows; absent classes get weight 1.",
                       reduction="Mean over valid target patches")
    for name, subset, counts in (("train", train_subset, train_counts), ("val", val_subset, val_counts)):
        print(f"{name}: {len(subset)} labeled windows; patch counts {dict(zip(classes, counts))}", flush=True)
        missing = [label for label, count in zip(classes, counts) if not count]
        if missing:
            print(f"  No {name} supervision for {missing}; their performance cannot be established.", flush=True)
    print(f"Focal loss: gamma={args.focal_gamma}, alpha={loss_config['alpha']}", flush=True)

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                       pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_subset, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), **loader_args)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_args)
    # Start with random weights. Fixed TRAIN z-scores live inside the encoder.
    encoder = PatchTST(patch_length=args.patch_length,
                       max_patches=args.window_size // args.patch_length,
                       train_stats=train.train_stats)
    model = PatchTSTPredictor(encoder, output_dim=len(classes), output_level="patch").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.output) if args.output else Path(__file__).parent / "checkpoints" / f"{args.target}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)  # Do not overwrite a previous run.
    config = {
        "arguments": vars(args), "encoder_config": encoder.config,
        "target": args.target, "classes": list(classes), "output_level": "patch",
        "target_layout": "batch,patch", "loss_logit_layout": "batch,patch,class",
        "patch_label_policy": {"rule": "unique_most_frequent_valid_class", "ties": "ignore",
                               "unknown_samples": "excluded", "all_unknown": "ignore", "minimum_valid_samples": 1},
        "train_statistics": train.train_stats,
        "manifest_sha256": hashlib.sha256((train.root / "manifest.json").read_bytes()).hexdigest(),
        "train_windows": len(train_subset), "val_windows": len(val_subset),
        "train_class_counts": train_counts, "val_class_counts": val_counts,
        "loss_config": loss_config,
        "selection_metric": "val.macro_f1", "metric_unit": "patch",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Training from scratch on {device}; saving to {output.resolve()}", flush=True)
    history, best_score = [], -1.0
    with ExitStack() as stack:
        writers = {}
        if args.tensorboard:
            for split in ("train", "val"):
                writer = SummaryWriter(log_dir=str(output / "tensorboard" / split))
                stack.callback(writer.close)
                writers[split] = writer
            print(f"Live epoch curves: tensorboard --logdir {output.resolve() / 'tensorboard'} "
                  "--host 127.0.0.1 --port 6006", flush=True)
        for epoch in range(1, args.epochs + 1):
            started = time.monotonic()
            train_metrics = run_epoch(model, train_loader, args.target, device, optimizer, **loss_args)
            val_metrics = run_epoch(model, val_loader, args.target, device, **loss_args)
            history.append({"epoch": epoch, "seconds": time.monotonic() - started,
                            "train": train_metrics, "val": val_metrics})
            for split, writer in writers.items():
                write_epoch_metrics(writer, epoch, history[-1][split])
            checkpoint = {"epoch": epoch, "config": config, "model_state": model.state_dict(),
                          "val_metrics": val_metrics}
            torch.save(checkpoint, output / "last.pt")
            if val_metrics["macro_f1"] > best_score:
                best_score = val_metrics["macro_f1"]
                torch.save(checkpoint, output / "best.pt")
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(f"Epoch {epoch}/{args.epochs} [{history[-1]['seconds']:.1f}s]", flush=True)
            for split, metrics in (("train", train_metrics), ("val", val_metrics)):
                scores = ", ".join(f"{name}={value['f1']:.3f}" for name, value in metrics["per_class"].items())
                if "binary_defect" in metrics:
                    scores += f", binary defect F1={metrics['binary_defect']['per_class']['defect']['f1']:.3f}"
                print(f"  {split:5s} loss={metrics['loss']:.4f}, macro F1={metrics['macro_f1']:.3f} "
                      f"(per-class F1: {scores})", flush=True)
    print(f"Best validation macro F1: {best_score:.3f}. Checkpoint: {output.resolve() / 'best.pt'}", flush=True)
    return output


if __name__ == "__main__":
    main()
