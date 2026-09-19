"""Load the instance-normalized ensemble without importing experiment runners.

Receipt checkpoint paths may be absolute (legacy runs) or relative to the JSON
receipt. Only load trusted PyTorch checkpoints; training files contain pickle.
"""
import argparse
import os
from pathlib import Path

import torch

from road_training.common import read, sha, write
from road_training.instance_model import InstancePatchTST, InstanceRoadModel
from road_training.streaming_model import Ensemble, StreamingRoadModel


def make_model(config):
    return InstanceRoadModel(InstancePatchTST(**config['encoder_config']),
                             **config['model_config'])


def checkpoint_files(receipt):
    """Resolve and verify every member before loading any model."""
    receipt = Path(receipt).resolve()
    members = read(receipt)['checkpoints']
    if not members:
        raise ValueError('An ensemble receipt must contain at least one checkpoint')
    paths = []
    for name, digest in members.items():
        path = Path(name)
        if not path.is_absolute():
            path = receipt.parent / path
        if sha(path) != digest:
            raise ValueError(f'Checkpoint SHA-256 mismatch: {path}')
        paths.append(path)
    return paths


def load_teachers(receipt, device='cuda'):
    models = []
    for path in checkpoint_files(receipt):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model = make_model(saved['config'])
        model.load_state_dict(saved['model_state'])
        models.append(model.to(device).eval())
    return Ensemble(models).to(device).eval()


def load_student(path, device='cuda'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = StreamingRoadModel(**saved['model_config']).to(device)
    model.load_state_dict(saved['model_state'])
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description='Write a portable ensemble receipt (no weights are copied).')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('checkpoints', type=Path, nargs='+')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new receipt path')
    paths = [p.resolve() for p in args.checkpoints]
    if len(set(paths)) != len(paths):
        parser.error('Duplicate checkpoint paths would change ensemble weighting')
    write(args.output, dict(checkpoints={os.path.relpath(p, args.output.resolve().parent): sha(p) for p in paths}))


if __name__ == '__main__':
    main()
