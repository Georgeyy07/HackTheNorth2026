"""A few training steps on TRAIN data, not a benchmark or full training run.

Run here: python model_example.py --source both --steps 5
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from road_training.dataset import RoadDataset
from road_training.patchtst import PatchTST, PatchTSTPretrainer, PatchTSTPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["real", "synthetic", "both"], default="both")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.manual_seed(17)
    torch.set_num_threads(2)
    data = RoadDataset(source=args.source, split="train", window_size=1024, stride=512)
    loader = DataLoader(data, batch_size=8, shuffle=True, num_workers=0)
    encoder = PatchTST(train_stats=data.train_stats)
    model = PatchTSTPretrainer(encoder, loss="l1").to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    iterator = iter(loader)
    losses = []
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        x, mask = batch["x"].to(args.device), batch["mask"].to(args.device)
        optimizer.zero_grad(set_to_none=True)
        result = model(x, mask)
        if not result["loss_mask"].any():
            continue
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(result["loss"].detach()))
        print(f"step {step+1}: masked L1 = {losses[-1]:.4f}", flush=True)

    # Transfer the SAME encoder and its saved normalizer to a new task head.
    # The four classes here are Kaggle's taxonomy; synthetic type labels have
    # three different classes. Your fine-tuning script selects the right target.
    classifier = PatchTSTPredictor(encoder, output_dim=4).to(args.device).eval()
    with torch.no_grad():
        logits = classifier(x, mask)
    print(json.dumps(dict(source=args.source, device=args.device,
                          train_recordings=len(data.records), pretraining_steps=len(losses),
                          pretrainer_parameters=sum(p.numel() for p in model.parameters()),
                          reconstruction_shape=list(result["prediction"].shape),
                          sample_classification_shape=list(logits.shape)), indent=2))


if __name__ == "__main__":
    main()
