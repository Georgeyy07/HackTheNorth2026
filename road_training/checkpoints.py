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
from road_training.acceleration_speed import CHANNELS as ACCELERATION_SPEED_CHANNELS
from road_training.dataset import CHANNELS as CANONICAL_CHANNELS


def channel_names(count):
    if count == 4:
        return list(ACCELERATION_SPEED_CHANNELS)
    if count == 7:
        return list(CANONICAL_CHANNELS)
    raise ValueError('Road inputs must have four or seven channels')


def model_channels(model):
    """Read the input width explicitly; never infer speed from the input array."""
    if hasattr(model, 'models'):
        counts = {model_channels(member) for member in model.models}
        if len(counts) != 1:
            raise ValueError('Ensemble members must use the same input channels')
        return counts.pop()
    encoder = getattr(model, 'encoder', model)
    if not hasattr(encoder, 'channels'):
        raise ValueError('Model must declare its input channels')
    count = encoder.channels
    channel_names(count)
    return count


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
    expected = read(receipt).get('input_channels')
    geometry = None
    for path in checkpoint_files(receipt):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model = make_model(saved['config'])
        names = channel_names(model.encoder.channels)
        if saved['config'].get('input_channels', names) != names or (expected is not None and expected != names):
            raise ValueError('Checkpoint and receipt input channels/order must match')
        current = (model.encoder.channels, model.encoder.patch_length, model.encoder.max_patches)
        if geometry is not None and geometry != current:
            raise ValueError('Ensemble members must use the same input channels and patch geometry')
        geometry = current
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
    members = {os.path.relpath(p, args.output.resolve().parent): sha(p) for p in paths}
    # Validate the contract before creating a receipt that other tools will trust.
    configs = [torch.load(p, map_location='cpu', weights_only=False)['config'] for p in paths]
    encoders = [make_model(c).encoder for c in configs]
    geometry = {(e.channels, e.patch_length, e.max_patches) for e in encoders}
    if len(geometry) != 1:
        parser.error('Ensemble members must use the same input channels and patch geometry')
    names = channel_names(encoders[0].channels)
    if any(c.get('input_channels', names) != names for c in configs):
        parser.error('Checkpoint input channel order does not match the model')
    write(args.output, dict(checkpoints=members, input_channels=names))


if __name__ == '__main__':
    main()
