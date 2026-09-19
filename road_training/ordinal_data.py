"""Small adapters for physical IRI classes and separate PVS weak labels."""
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from road_training.common import read


class PVSDataset(Dataset):
    def __init__(self, root, *, window_size=1024, stride=1):
        self.root = Path(root)
        manifest = read(self.root/'manifest.json')
        if manifest['channels'] != ['accel_x', 'accel_y', 'accel_z', 'speed'] or manifest['sample_rate_hz'] != 100:
            raise ValueError('Expected prepared four-input 100-Hz PVS data')
        if window_size <= 0 or stride <= 0:
            raise ValueError('Positive window size and stride required')
        self.window_size, self.stride = window_size, stride
        self.records, self.ends, self.arrays = [], [], []
        count = 0
        for record in manifest['records']:
            if record['split'] != 'train':
                raise ValueError('PVS auxiliary data must be TRAIN-only')
            windows = max(0, (record['samples']-window_size)//stride+1)
            if not windows:
                continue
            self.records.append(record)
            self.arrays.append({k: np.load(self.root/record['path']/f'{k}.npy', mmap_mode='r')
                                for k in ('x', 'mask', 'quality')})
            count += windows
            self.ends.append(count)
        if not count:
            raise ValueError('No complete PVS windows')

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        record = bisect_right(self.ends, index)
        offset = self.ends[record-1] if record else 0
        start = (index-offset)*self.stride
        result = {k: torch.from_numpy(np.array(v[start:start+self.window_size], copy=True))
                  for k, v in self.arrays[record].items()}
        result.update(vehicle=self.records[record]['vehicle'], recording_id=self.records[record]['id'])
        return result


def pvs_patch_targets(batch, patch_length=16):
    """Require a fully observed patch with one unambiguous weak class."""
    labels = batch['quality'].long()
    if labels.shape[1] % patch_length:
        raise ValueError('Expected whole patches')
    labels = labels.reshape(len(labels), -1, patch_length)
    observed = batch['mask'][..., :3].all(-1).reshape_as(labels)
    valid = observed.all(-1) & (labels.amin(-1) >= 0) & (labels.amin(-1) == labels.amax(-1))
    return labels[..., 0].masked_fill(~valid, -100), valid
