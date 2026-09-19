"""Attach same-recording past and bounded future context to ordinary windows."""
from bisect import bisect_right
import numpy as np
import torch
from torch.utils.data import Dataset


class ContextWindows(Dataset):
    def __init__(self, base, context_patches=48, delay_patches=2, patch_length=16):
        self.base = base
        self.context = context_patches*patch_length
        self.future = delay_patches*patch_length

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        batch = self.base[index]
        record_index = bisect_right(self.base._ends, index)
        arrays = self.base._open(record_index)
        lo = batch['start']-self.context
        hi = batch['start']+self.base.window_size+self.future
        left, right = max(lo, 0), min(hi, len(arrays['x']))
        ends = lo + np.arange(1, (hi-lo)//16+1)*16
        batch['context_available'] = torch.from_numpy((ends <= len(arrays['x'])) & (ends > 0))
        for key in ('x', 'mask'):
            values = np.zeros((hi-lo, 7), dtype=arrays[key].dtype)
            values[left-lo:right-lo] = arrays[key][left:right]
            batch['context_'+key] = torch.from_numpy(values)
        return batch
