"""Evaluate a frozen two-head checkpoint on complete non-overlapping windows.

All normalization and loss settings come from the checkpoint. No fitting,
threshold selection or checkpoint selection occurs here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__:
    from road_training.dataset import RoadDataset
    from road_training.patchtst import PatchTST, PatchTSTRoadModel
    from road_training.train_multitask import run_epoch, supervised_windows
else:
    from road_training.dataset import RoadDataset
    from road_training.patchtst import PatchTST, PatchTSTRoadModel
    from road_training.train_multitask import run_epoch, supervised_windows


def evaluate(checkpoint, *, data_root=None, split="test", source="real",
             real_dataset="all", batch_size=256, device="cuda", precision="bf16",
             num_workers=0):
    checkpoint=Path(checkpoint)
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    config=saved["config"]
    if config["model"]!="PatchTSTRoadModel":
        raise ValueError("Expected a two-head PatchTSTRoadModel checkpoint")
    window=config["arguments"]["window_size"]
    dataset=RoadDataset(data_root,split=split,source=source,real_dataset=real_dataset,
                        window_size=window,stride=window,return_labels=True)
    manifest_sha=hashlib.sha256((dataset.root/"manifest.json").read_bytes()).hexdigest()
    if manifest_sha!=config["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the training checkpoint")
    encoder=PatchTST(**config["encoder_config"])
    model=PatchTSTRoadModel(encoder)
    model.load_state_dict(saved["model_state"])
    # Scalers belong to the training checkpoint, not the evaluation filter.
    torch.testing.assert_close(encoder.mean,torch.tensor(config["train_statistics"]["mean"],dtype=torch.float32))
    torch.testing.assert_close(encoder.std,torch.tensor(config["train_statistics"]["std"],dtype=torch.float32))
    model.to(device)
    selected,_,_=supervised_windows(dataset,encoder.patch_length)
    loader=DataLoader(selected,batch_size=batch_size,num_workers=num_workers,
                      shuffle=False,pin_memory=torch.device(device).type=="cuda")
    loss=config["loss"]
    metrics=run_epoch(model,loader,device,precision=precision,gamma=loss["gamma"],
        alpha=torch.tensor(loss["alpha"],device=device),
        roughness_weight=loss["roughness_weight"],disturbance_weight=loss["disturbance_weight"])
    return dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        epoch=saved["epoch"],split=split,source=source,real_dataset=real_dataset,
        windows=len(selected),window_stride=window,manifest_sha256=manifest_sha,
        disturbance_probability_threshold=.5,normalization="Frozen TRAIN checkpoint statistics",
        metrics=metrics)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data-root")
    p.add_argument("--split",choices=("train","val","test"),default="test")
    p.add_argument("--source",choices=("real","synthetic","both"),default="real")
    p.add_argument("--real-dataset",choices=("all","kaggle","lira","roadsens"),default="all")
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--device",default="cuda")
    p.add_argument("--precision",choices=("bf16","fp32"),default="bf16")
    p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    if args.batch_size<1 or args.num_workers<0:p.error("Invalid batch-size or num-workers")
    result=evaluate(**{key:value for key,value in vars(args).items() if key!="output"})
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as handle:handle.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=="__main__":main()
