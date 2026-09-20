"""Train all three real datasets with sensor augmentation, then add MixUp.

Run --init once to freeze the plan, then --run. All runs finish before TEST is
opened. Live epoch metrics and TensorBoard contain clean TRAIN and real VAL.
"""
import argparse
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

from road_training.common import read, write, sha
from road_training.dataset import RoadDataset, collection_name
from road_training.patchtst import PatchTST, PatchTSTRoadModel
from road_training.train import balanced_alpha
from road_training.train_multitask import patch_targets, run_epoch, supervised_windows, epoch_progress, write_metrics
from road_training.augmentation import perturb
from road_training.mixup import mix_disturbance, mixed_joint_loss, latent_forward

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/real_roadsens_mixup_20260917"
DATA = ROOT / "road_training/data_with_roadsens"


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "plan.json").exists():
        raise FileExistsError("Preserve the existing study plan")
    paths = [Path(__file__), ROOT/"road_training/mixup.py", ROOT/"road_training/dataset.py",
             ROOT/"road_training/patchtst.py", ROOT/"road_training/train.py",
             ROOT/"road_training/train_multitask.py", ROOT/"road_training/augmentation.py",
             ROOT/"road_training/evaluate_multitask.py", ROOT/"road_training/metrics.py",
             ROOT/"road_training/common.py",
             ROOT/"reports/mixup_time_series_research_20260917/report.md"]
    plan = dict(arms=["sensor_aug", "sensor_aug_mixup", "sensor_aug_latent_mixup"], seeds=[42, 43, 44],
        run_order="Finish sensor_aug seeds first, then input MixUp, then latent MixUp; TEST after all nine checkpoints freeze",
        epochs=12, steps_per_epoch=512, batch_size=256, lr=1e-4, weight_decay=.01,
        dropout=.1, precision="bf16", device="cuda", window_size=1024, patch_length=16,
        source="real", samples_per_batch=dict(kaggle=64, lira=128, roadsens=64),
        sampling="Uniform eligible stride-one windows within each dataset, with replacement; matched sequence across arms",
        epoch_definition="512 sampled batches (131072 window draws), not one exhaustive pass over overlapping windows",
        early_stopping=dict(minimum_updates=1024, patience_epochs=6, minimum_improvement=1e-4),
        selection="Minimum combined real-VAL Huber + focal loss; fixed detection threshold 0.5",
        sensor_augmentation=dict(rotation_max_degrees=5., accel_noise_std=.005, accel_bias_bound=.02,
            gyro_noise_std=.0002, gyro_bias_bound=.001, noise_policy="Clipped at 3 SD; identical held vectors share noise"),
        mixup=dict(beta_alpha=.2, probability=.5, datasets=["kaggle", "roadsens"],
            input_policy="Same dataset and available channels; masks intersect; original timing retained",
            target_policy="Weighted hard-label focal terms only where BOTH patch targets are known; LiRA/IRI never mixed",
            latent_policy="Mix corresponding encoder patch features before disturbance head; keep roughness features unchanged",
            retained_originals="Half eligible rows remain unmixed in expectation; no extra forward/optimizer steps",
            references=["https://arxiv.org/abs/1710.09412", "https://arxiv.org/abs/2304.04271",
                        "https://arxiv.org/abs/2309.09970", "https://arxiv.org/abs/2309.13439"],
            research_report=str(ROOT/"reports/mixup_time_series_research_20260917/report.md")),
        loss=dict(gamma=2., roughness_weight=1., disturbance_weight=1., alpha_fit="Non-overlapping real TRAIN patches"),
        data_root=str(DATA), manifest_sha256=sha(DATA/"manifest.json"),
        source_sha256={str(p): sha(p) for p in paths},
        limitations=["RoadSens is TRAIN-only; no RoadSens held-out performance claim.",
            "Existing TEST has historical exposure; no new external road/device validation.",
            "MixUp is a statistical regularizer and does not simulate physical road roughness.",
            "Three seeds measure optimizer variability on the same roads.",
            "The fixed dataset proportions differ from earlier uniform-window studies."])
    write(OUT/"plan.json", plan)
    with tarfile.open(OUT/"sources.tar.gz", "w:gz") as archive:
        for p in paths:
            archive.add(p, arcname=str(p.relative_to(ROOT)))
    return plan


def check_plan():
    plan = read(OUT/"plan.json")
    if sha(DATA/"manifest.json") != plan["manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    for path, expected in plan["source_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen training source changed: {path}")
    return plan


def dataset_pools(data, indices):
    if data.source != "real" or data.split != "train":
        raise ValueError("Sampling must use real TRAIN only")
    ids = np.asarray(indices, dtype=np.int64)
    record_index = np.searchsorted(data._ends, ids, side="right")
    names = np.array([collection_name(r) for r in data.records])
    return {name: ids[names[record_index] == name] for name in ("kaggle", "lira", "roadsens")}


def sample(pools, counts, seed, epoch, steps):
    rng = np.random.default_rng([seed, epoch, 814])
    draws = [rng.choice(pools[name], (steps, count), replace=True) for name, count in counts.items()]
    return np.concatenate(draws, axis=1).reshape(-1).tolist()


def train_epoch(model, loader, optimizer, alpha, *, seed, epoch, mixup, plan, progress):
    """Log augmented objective; measure F1 separately on unmixed clean inputs."""
    model.train()
    rsum = dsum = nr = nd = updates = mixed_windows = before_known = after_known = 0
    for step, batch in enumerate(loader):
        targets = {k: v.cuda(non_blocking=True) for k, v in patch_targets(batch, 16).items()}
        x, mask = batch["x"].cuda(non_blocking=True), batch["mask"].cuda(non_blocking=True)
        global_step = (epoch-1)*plan["steps_per_epoch"]+step
        x = perturb(x, mask, seed=seed, step=global_step, rotate=True, noise=True)
        recipe = None
        if mixup:
            mixed_x, mixed_mask, recipe = mix_disturbance(x, mask, targets, batch["dataset"], seed=seed,
                step=global_step, alpha=plan["mixup"]["beta_alpha"], probability=plan["mixup"]["probability"])
            mixed_windows += int(recipe["active"].sum())
            if mixup != "latent":
                x, mask = mixed_x, mixed_mask
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = latent_forward(model, x, mask, recipe) if mixup == "latent" else model(x, mask)
        losses = mixed_joint_loss(output, targets, recipe, gamma=2., alpha=alpha)
        nrough, ndist = int(losses["roughness_valid"].sum()), int(losses["disturbance_valid"].sum())
        if not nrough+ndist or not torch.isfinite(losses["total"]):
            raise FloatingPointError("Batch has no usable objective")
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        rsum += losses["roughness"].item()*nrough
        dsum += losses["disturbance"].item()*ndist
        nr += nrough; nd += ndist; updates += 1
        before_known += int(targets["disturbance_valid"].sum()); after_known += ndist
        progress(step+1, len(loader))
    return dict(loss=rsum/max(nr, 1)+dsum/max(nd, 1), roughness_loss=rsum/max(nr, 1),
                disturbance_loss=dsum/max(nd, 1), roughness_patches=nr, disturbance_patches=nd,
                optimizer_updates=updates, mixed_windows=mixed_windows,
                disturbance_patches_before_mixing=before_known, disturbance_patches_after_mixing=after_known)


def save(path, checkpoint):
    pending = path.with_suffix(".pending.pt")
    torch.save(checkpoint, pending)
    pending.replace(path)


def train(arm, seed):
    plan = check_plan()
    folder = OUT/f"{arm}_seed{seed}"
    if (folder/"complete.json").exists():
        assert sha(folder/"best.pt") == read(folder/"complete.json")["checkpoint_sha256"]
        return
    folder.mkdir(exist_ok=True)
    if arm not in plan["arms"] or seed not in plan["seeds"]:
        raise ValueError("Run is not declared")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data = RoadDataset(DATA, source="real", split="train", stride=1, return_labels=True, max_cached_recordings=128)
    selected, counts, _ = supervised_windows(data, 16)
    pools = dataset_pools(data, selected.indices)
    if any(not len(pool) for pool in pools.values()) or any(r["source"] != "real" for r in data.records):
        raise ValueError("Expected all three real TRAIN datasets, no synthetic examples")
    val = RoadDataset(DATA, source="real", split="val", stride=1024, return_labels=True)
    validation, _, _ = supervised_windows(val, 16)
    alpha = balanced_alpha(counts).cuda()
    model = PatchTSTRoadModel(PatchTST(max_patches=64, train_stats=data.train_stats)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan["lr"], weight_decay=plan["weight_decay"])
    config = dict(model="PatchTSTRoadModel", arm=arm, seed=seed, data_root=str(DATA), source="real",
        manifest_sha256=plan["manifest_sha256"], plan_sha256=sha(OUT/"plan.json"),
        encoder_config=model.encoder.config, train_statistics=data.train_stats,
        parameters=sum(p.numel() for p in model.parameters()),
        arguments={key: plan[key] for key in ("epochs", "steps_per_epoch", "batch_size", "lr", "weight_decay",
                    "window_size", "precision", "device")},
        loss=dict(gamma=2., alpha=alpha.cpu().tolist(), roughness_weight=1., disturbance_weight=1.),
        train_windows_by_dataset={name: len(pool) for name, pool in pools.items()},
        samples_per_batch=plan["samples_per_batch"], train_patch_counts=counts,
        selection_metric="val.loss", augmentation=plan["sensor_augmentation"],
        mixup=plan["mixup"] if arm.endswith("mixup") else None,
        train_recordings=[r["id"] for r in data.records], val_recordings=[r["id"] for r in val.records])
    if (folder/"config.json").exists() and read(folder/"config.json") != config:
        raise ValueError("Resume configuration changed")
    write(folder/"config.json", config)
    history, best, significant, stale = [], float("inf"), float("inf"), 0
    if (folder/"last.pt").exists():
        checkpoint = torch.load(folder/"last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"]); optimizer.load_state_dict(checkpoint["optimizer_state"])
        history, best, significant, stale = (checkpoint[k] for k in ("history", "best_loss", "significant_best", "stale"))
        torch.set_rng_state(checkpoint["torch_rng"]); torch.cuda.set_rng_state(checkpoint["cuda_rng"])
    loader_args = dict(batch_size=256, num_workers=2, pin_memory=True)
    val_loader = DataLoader(validation, shuffle=False, generator=torch.Generator().manual_seed(917), **loader_args)
    probe_indices = sample(pools, plan["samples_per_batch"], 917, 0, 4)
    probe = DataLoader(Subset(data, probe_indices), shuffle=False, generator=torch.Generator().manual_seed(918), **loader_args)
    loss_args = dict(precision="bf16", gamma=2., alpha=alpha)
    early = plan["early_stopping"]
    start = time.monotonic()
    loader = None  # A resumed run may already have finished its last epoch.
    with SummaryWriter(str(folder/"tensorboard")) as writer:
        for epoch in range(len(history)+1, plan["epochs"]+1):
            if history and history[-1]["updates"] >= early["minimum_updates"] and stale >= early["patience_epochs"]:
                break
            indices = sample(pools, plan["samples_per_batch"], seed, epoch, plan["steps_per_epoch"])
            loader = DataLoader(Subset(data, indices), shuffle=False,
                generator=torch.Generator().manual_seed(seed*1000+epoch), **loader_args)
            objective = train_epoch(model, loader, optimizer, alpha, seed=seed, epoch=epoch,
                mixup="latent" if "latent" in arm else arm.endswith("mixup"), plan=plan,
                progress=epoch_progress(folder, epoch, plan["epochs"], "train"))
            assert objective["optimizer_updates"] == plan["steps_per_epoch"]
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                scores = run_epoch(model, val_loader, "cuda", **loss_args)
                clean = run_epoch(model, probe, "cuda", **loss_args)
            row = dict(epoch=epoch, updates=epoch*plan["steps_per_epoch"], objective=objective,
                       train_clean=clean, val=scores, elapsed_seconds=time.monotonic()-start)
            history.append(row)
            improved = scores["loss"] < best
            best = min(best, scores["loss"])
            if scores["loss"] < significant-early["minimum_improvement"]:
                significant, stale = scores["loss"], 0
            else:
                stale += 1
            checkpoint = dict(epoch=epoch, config=config, model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(), history=history, val_metrics=scores,
                best_loss=best, significant_best=significant, stale=stale,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
            save(folder/"last.pt", checkpoint)
            if improved:
                save(folder/"best.pt", checkpoint)
            write(folder/"history.json", history)
            for prefix, values in [("train_clean/", clean), ("val/", scores)]:
                write_metrics(writer, epoch, values, prefix)
            writer.add_scalar("train_augmented/loss", objective["loss"], epoch); writer.flush()
            f1 = scores["disturbance"]["classification"]["per_class"]["disturbance"]["f1"]
            print(f"{arm} seed={seed} epoch={epoch}: VAL F1={f1:.4f}, IRI patch MAE={scores['roughness']['mae']:.4f}, loss={scores['loss']:.4f}", flush=True)
            write(folder/"progress.json", dict(state="running", epoch=epoch, updates=row["updates"],
                best_loss=best, stale=stale, val_f1=f1, val_iri_mae=scores["roughness"]["mae"]))
    best_checkpoint = torch.load(folder/"best.pt", map_location="cpu", weights_only=False)
    write(folder/"complete.json", dict(checkpoint_sha256=sha(folder/"best.pt"), epochs=len(history),
        updates=history[-1]["updates"], best_epoch=best_checkpoint["epoch"], best_loss=best,
        test_evaluated=False, elapsed_seconds=time.monotonic()-start))
    write(folder/"progress.json", dict(state="complete", **read(folder/"complete.json")))
    del loader, val_loader, probe, model, optimizer
    gc.collect(); torch.cuda.empty_cache()


def report_results(plan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for arm in plan["arms"]:
        for seed in plan["seeds"]:
            folder = OUT/f"{arm}_seed{seed}"
            for split in ("val", "test"):
                patches = read(folder/f"{split}_patches.json")
                details = read(folder/f"{split}_details.json")
                binary = patches["metrics"]["disturbance"]["classification"]["per_class"]["disturbance"]
                rows.append(dict(arm=arm, seed=seed, split=split, epoch=patches["epoch"],
                    **{key: binary[key] for key in ("f1", "precision", "recall")},
                    section_iri_mae=details["roughness"]["mae"],
                    ordinal_macro_f1=details["roughness"]["ordinal"]["macro_f1_present_classes"],
                    average_precision=details["disturbance"][0]["patch_ranking"]["average_precision"]))
    summary = []
    for arm in plan["arms"]:
        for split in ("val", "test"):
            selected = [r for r in rows if r["arm"] == arm and r["split"] == split]
            means = {key: float(np.mean([r[key] for r in selected])) for key in
                     ("f1", "precision", "recall", "section_iri_mae", "ordinal_macro_f1", "average_precision")}
            deviations = {key: float(np.std([r[key] for r in selected], ddof=1)) for key in means}
            summary.append(dict(arm=arm, split=split, mean=means, std=deviations))
    paired = []
    for split in ("val", "test"):
        for seed in plan["seeds"]:
            a = next(r for r in rows if r["arm"] == "sensor_aug" and r["seed"] == seed and r["split"] == split)
            for arm in plan["arms"][1:]:
                b = next(r for r in rows if r["arm"] == arm and r["seed"] == seed and r["split"] == split)
                paired.append(dict(arm=arm, seed=seed, split=split, f1=b["f1"]-a["f1"], section_iri_mae=b["section_iri_mae"]-a["section_iri_mae"]))
    write(OUT/"results.json", dict(rows=rows, summary=summary, mixup_minus_sensor_aug=paired))
    lines = ["# All-real sensor augmentation and MixUp", "", "All nine runs and frozen-checkpoint evaluations completed.", "",
        "TRAIN: Kaggle + LiRA + RoadSens IMU only. VAL/TEST: original held-out Kaggle and LiRA roads. RoadSens has no held-out split.", "",
        "| Training | Split | Disturbance F1 | Precision | Recall | Section IRI MAE | Ordinal macro F1 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for item in summary:
        a = item["mean"]
        lines.append(f"| {item['arm']} | {item['split']} | {a['f1']:.3f} | {a['precision']:.3f} | {a['recall']:.3f} | {a['section_iri_mae']:.3f} | {a['ordinal_macro_f1']:.3f} |")
    lines += ["", "Means over three seeds. F1 is binary disturbance F1; IRI is m/km (lower is better). Ordinal scores use existing IRI bins and only classes represented on the held-out roads.", "",
        "All arms use shared ≤5° IMU rotation, small held-sample noise and constant sensor bias. Input MixUp blends Kaggle/RoadSens windows within their dataset; latent MixUp blends encoder features at corresponding patches. Both use an expected hard-label focal loss and both-label-known masking. LiRA roughness examples remain unmixed.", "",
        "Every 256-example batch contains 64 Kaggle, 128 LiRA and 64 RoadSens windows. Window draws, initialization and sensor perturbations match across paired seeds. Unknown labels and missing channels are never invented. Epochs comprise 512 sampled batches; early stopping and checkpoint selection use only real validation loss.", "",
        "No augmentation is applied to validation/test. Clean TRAIN probe metrics use the same 1,024 fixed windows with dropout disabled; they are not whole-training-set scores. TEST was evaluated only after all selected checkpoint hashes were frozen. Existing TEST has historical exposure, and three seeds do not establish generalization to new roads or devices.", "",
        "This comparison isolates adding MixUp to the new all-real training recipe. It does not independently isolate the benefit of RoadSens against the older dataset/sampling protocols.", "",
        "[Paired results](results.json) · [Frozen plan](plan.json) · [Learning curves](learning_curves.png)", "",
        "[Literature review and implementation rationale](../mixup_time_series_research_20260917/report.md). Our masking, within-dataset pairing and focal-loss treatment are adaptations for partial road-sensor labels; this is not an exact reproduction of MixUp++."]
    (OUT/"report.md").write_text("\n".join(lines)+"\n")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for j, seed in enumerate(plan["seeds"]):
        for arm, color in zip(plan["arms"], ("tab:blue", "tab:orange", "tab:green")):
            history = read(OUT/f"{arm}_seed{seed}"/"history.json")
            for phase, style in [("val", "-"), ("train_clean", "--")]:
                x = [r["updates"] for r in history]
                axes[0, j].plot(x, [r[phase]["by_dataset"]["kaggle"]["disturbance"]["classification"]["per_class"]["disturbance"]["f1"] for r in history], style, color=color, label=f"{arm} {phase}")
                axes[1, j].plot(x, [r[phase]["roughness"]["mae"] for r in history], style, color=color)
        axes[0, j].set_title(f"Seed {seed}")
        axes[1, j].set_xlabel("Optimizer updates")
    axes[0, 0].set_ylabel("Kaggle disturbance F1"); axes[1, 0].set_ylabel("Patch IRI MAE (m/km)")
    axes[0, 0].legend(fontsize=7)
    fig.savefig(OUT/"learning_curves.png", dpi=160); plt.close(fig)


def run_suite():
    from road_training.evaluate_multitask import evaluate as patches
    from road_training.metrics import evaluate as details
    plan = check_plan()
    jobs = [(arm, seed) for arm in plan["arms"] for seed in plan["seeds"]]
    completed = []
    for arm, seed in jobs:
        write(OUT/"progress.json", dict(state="training", pid=os.getpid(), active=dict(arm=arm, seed=seed),
              completed=completed, total=len(jobs)))
        train(arm, seed)
        completed.append(dict(arm=arm, seed=seed))
    checkpoints = {f"{arm}_seed{seed}": sha(OUT/f"{arm}_seed{seed}"/"best.pt") for arm, seed in jobs}
    frozen = dict(plan_sha256=sha(OUT/"plan.json"), checkpoints=checkpoints, selection="Only real VAL loss")
    if (OUT/"evaluation_plan.json").exists() and read(OUT/"evaluation_plan.json") != frozen:
        raise ValueError("Frozen evaluation checkpoints changed")
    write(OUT/"evaluation_plan.json", frozen)
    for arm, seed in jobs:
        folder = OUT/f"{arm}_seed{seed}"
        for split in ("val", "test"):
            check_plan()
            assert sha(folder/"best.pt") == checkpoints[folder.name]
            write(OUT/"progress.json", dict(state="evaluating", pid=os.getpid(), arm=arm, seed=seed, split=split))
            p, d = folder/f"{split}_patches.json", folder/f"{split}_details.json"
            if not p.exists():
                write(p, patches(folder/"best.pt", data_root=DATA, split=split, source="real", num_workers=2))
            if not d.exists():
                details(folder/"best.pt", d, split=split)
            assert read(p)["checkpoint_sha256"] == read(d)["checkpoint_sha256"] == checkpoints[folder.name]
    report_results(plan)
    write(OUT/"progress.json", dict(state="complete", pid=os.getpid(), completed=completed,
                                    evaluation_plan_sha256=sha(OUT/"evaluation_plan.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.init:
        initialize()
    elif args.run:
        import fcntl
        with (OUT/"runner.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run_suite()
            except Exception:
                write(OUT/"failure.json", dict(error=traceback.format_exc(), pid=os.getpid()))
                write(OUT/"progress.json", dict(state="failed", pid=os.getpid()))
                raise
    else:
        parser.error("Use --init or --run")
