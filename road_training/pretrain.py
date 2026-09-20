"""Simple DropPatch / ArcTan patch-reconstruction trainer.

Run from the repository root: python -m road_training.pretrain --help
Uses TRAIN for gradients, VAL for reconstruction checkpoint selection, no TEST.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from road_training.arctan import ArcTanEncoder, ArcTanPretrainer
from road_training.dataset import RoadDataset
from road_training.droppatch import DropPatchPretrainer
from road_training.patchtst import PatchTST


def build_pretrainer(method, encoder_config, pretraining_config=None, train_stats=None):
    """Explicit factory; saved configurations use these same constructors."""
    if method == "droppatch":
        encoder = PatchTST(**encoder_config, train_stats=train_stats)
        return DropPatchPretrainer(encoder, **(pretraining_config or {}))
    if method == "arctan":
        encoder = ArcTanEncoder(**encoder_config, train_stats=train_stats)
        return ArcTanPretrainer(encoder, **(pretraining_config or {}))
    raise ValueError("method must be 'droppatch' or 'arctan'")


def load_pretrained_encoder(path, device="cpu"):
    """Load encoder + TRAIN normalizer only; attach fresh downstream heads.

    ArcTan also retains its timestep embedding and Eq. (2) latent projection.
    Its ordinary forward() uses clean t=1. Decoder weights are not transferred.
    """
    saved = torch.load(path, map_location="cpu", weights_only=True)
    kind = {"droppatch": PatchTST, "arctan": ArcTanEncoder}[saved["method"]]
    encoder = kind(**saved["encoder_config"])
    encoder.load_state_dict(saved["encoder_state"], strict=True)
    return encoder.to(device)


def run_epoch(model, loader, device, generator, optimizer=None, scheduler=None):
    """Average by observed target scalars, not by unequal batch means."""
    training = optimizer is not None
    model.train(training)
    loss_sum = mse_sum = 0.
    target_count = updates = skipped = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with autocast:
                result = model(x, mask, generator=generator)
            count = int(result["loss_mask"].sum())
            if not count:
                skipped += 1
                continue
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite pretraining loss")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                updates += 1
            with torch.no_grad():
                selected = result["loss_mask"]
                residual = result["prediction"].float()[selected] - result["target"].float()[selected]
                loss_sum += float(loss.detach()) * count
                mse_sum += float(residual.square().sum())
                target_count += count
    if not target_count:
        raise ValueError("No observed reconstruction targets in this epoch")
    return dict(loss=loss_sum / target_count, reconstruction_mse=mse_sum / target_count,
                target_values=target_count, optimizer_updates=updates, skipped_batches=skipped)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["droppatch", "arctan"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data_transfer_v2_mount")
    parser.add_argument("--source", choices=["real", "synthetic", "both"], default="both")
    parser.add_argument("--val-source", choices=["real", "synthetic", "both"], default=None)
    parser.add_argument("--real-dataset", choices=["all", "kaggle", "lira", "roadsens"], default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--stop-after-epochs", type=int, default=None,
                        help="Stop early without changing the original cosine schedule horizon")
    parser.add_argument("--steps-per-epoch", type=int, default=512,
                        help="Random TRAIN batches per epoch, sampled with replacement")
    parser.add_argument("--val-windows", type=int, default=4096, help="Evenly spaced VAL windows; 0 means all")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--patch-length", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--drop-ratio", type=float, default=.6)
    parser.add_argument("--mask-ratio", type=float, default=.4)
    parser.add_argument("--channel-mode", choices=["mixed", "independent"], default="mixed")
    parser.add_argument("--timestep-mode", choices=["token", "sequence"], default="token")
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    positive = (args.epochs, args.steps_per_epoch, args.batch_size, args.window_size,
                args.stride, args.patch_length, args.d_model)
    if min(positive) < 1 or args.val_windows < 0 or args.workers < 0 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("Invalid training dimensions, counts, learning rate or weight decay")
    if args.window_size % args.patch_length:
        parser.error("--window-size must be divisible by --patch-length")
    if args.stop_after_epochs is not None and not 1 <= args.stop_after_epochs <= args.epochs:
        parser.error("--stop-after-epochs must be between 1 and --epochs")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        parser.error("CUDA with BF16 is required by default; use --device cpu explicitly for CPU tests")
    if device.type not in ("cuda", "cpu"):
        parser.error("Supported devices: CUDA or CPU")
    if args.output.exists():
        parser.error("--output already exists; choose a new directory to preserve previous runs")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    data_args = dict(root=args.data_root, window_size=args.window_size, return_labels=False,
                     real_dataset=args.real_dataset)
    train = RoadDataset(**data_args, split="train", source=args.source, stride=args.stride)
    val = RoadDataset(**data_args, split="val", source=args.val_source or args.source, stride=args.window_size)
    val_indices = list(range(len(val)))
    if args.val_windows and len(val) > args.val_windows:
        val_indices = torch.linspace(0, len(val) - 1, args.val_windows).long().tolist()
    n_layers = args.n_layers if args.n_layers is not None else (3 if args.method == "droppatch" else 2)
    n_heads = args.n_heads if args.n_heads is not None else (4 if args.method == "droppatch" else 16)
    encoder_config = dict(channels=7, patch_length=args.patch_length, d_model=args.d_model,
                          n_heads=n_heads, n_layers=n_layers, ffn_dim=4 * args.d_model,
                          dropout=args.dropout, max_patches=max(256, args.window_size // args.patch_length))
    if args.method == "droppatch":
        pretraining_config = dict(drop_ratio=args.drop_ratio, mask_ratio=args.mask_ratio)
    else:
        encoder_config["channel_mode"] = args.channel_mode
        pretraining_config = dict(decoder_layers=args.decoder_layers, timestep_mode=args.timestep_mode)
    model = build_pretrainer(args.method, encoder_config, pretraining_config, train.train_stats).to(device)
    sampling_rng = torch.Generator().manual_seed(args.seed + 1)
    sampler = RandomSampler(train, replacement=True, num_samples=args.steps_per_epoch * args.batch_size,
                            generator=sampling_rng)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, sampler=sampler, **loader_args,
                              generator=torch.Generator().manual_seed(args.seed + 2))
    val_loader = DataLoader(Subset(val, val_indices), shuffle=False, **loader_args,
                            generator=torch.Generator().manual_seed(args.seed + 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * args.steps_per_epoch)
    corruption_rng = torch.Generator(device=device).manual_seed(args.seed + 4)
    config = dict(method=args.method, encoder_config=model.encoder.config,
                  pretraining_config=model.config, train_statistics=train.train_stats,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  manifest_sha256=hashlib.sha256((train.root / "manifest.json").read_bytes()).hexdigest(),
                  train_recordings=[r["path"] for r in train.records],
                  val_recordings=[r["path"] for r in val.records], val_indices=val_indices,
                  train_windows=len(train), val_windows=len(val_indices),
                  parameters=sum(p.numel() for p in model.parameters()),
                  encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
                  selection="minimum VAL reconstruction objective, fixed corruption RNG, no TEST")
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    history, best = [], float("inf")
    for epoch in range(1, (args.stop_after_epochs or args.epochs) + 1):
        started = time.monotonic()
        train_metrics = run_epoch(model, train_loader, device, corruption_rng, optimizer, scheduler)
        # Reset only the local corruption generator: every checkpoint sees the
        # same VAL drop/mask/noise/timestep draws without changing TRAIN RNG.
        validation_rng = torch.Generator(device=device).manual_seed(args.seed + 5)
        val_metrics = run_epoch(model, val_loader, device, validation_rng)
        row = dict(epoch=epoch, train=train_metrics, val=val_metrics,
                   lr=optimizer.param_groups[0]["lr"], seconds=time.monotonic() - started)
        history.append(row)
        saved = dict(method=args.method, encoder_config=model.encoder.config,
                     pretraining_config=model.config, model_state=model.state_dict(),
                     encoder_state=model.encoder.state_dict(), epoch=epoch, metrics=row,
                     manifest_sha256=config["manifest_sha256"])
        torch.save(saved, args.output / "last.pt")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            torch.save(saved, args.output / "best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch {epoch:3d} | train loss {train_metrics['loss']:.6f} | "
              f"val loss {val_metrics['loss']:.6f} | val patch MSE {val_metrics['reconstruction_mse']:.6f}", flush=True)


if __name__ == "__main__":
    main()
