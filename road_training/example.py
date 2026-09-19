"""Run from this directory: python example.py --source both"""
import argparse

import torch
from torch.utils.data import DataLoader

from road_training.dataset import RoadDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["real", "synthetic", "both"], default="both")
    args = parser.parse_args()

    # 65 patches: 64 input patches and their 64 next-patch targets.
    # These shapes require a CAUSAL model. For the bidirectional PatchTST
    # encoder, use masked reconstruction as shown in model_example.py.
    dataset = RoadDataset(source=args.source, split="train", window_size=65 * 16, stride=512)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    batch = next(iter(loader))
    mean = torch.tensor(dataset.train_stats["mean"], dtype=torch.float32)
    std = torch.tensor(dataset.train_stats["std"], dtype=torch.float32)
    x = ((batch["x"] - mean) / std).masked_fill(~batch["mask"], 0)
    patches = x.reshape(len(x), 65, 16, 7)
    patch_mask = batch["mask"].reshape_as(patches)
    inputs, targets = patches[:, :-1], patches[:, 1:]
    target_mask = patch_mask[:, 1:]
    print(f"{len(dataset):,} windows from {len(dataset.records)} recordings")
    print(f"Inputs: {tuple(inputs.shape)}; targets: {tuple(targets.shape)}")
    print(f"Valid target values: {target_mask.float().mean():.2%}")


if __name__ == "__main__":
    main()
