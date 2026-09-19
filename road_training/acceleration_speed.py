"""Four-channel data and augmentation for acceleration XYZ plus speed.

The neural network is the existing InstanceRoadModel with encoder channels=4.
Gyroscope values and masks are removed before training or normalization.
"""
import torch
from torch import nn

from .dataset import RoadDataset
from .mounting_augmentation import perturb as imu_augmentation


CHANNELS = ('accel_x', 'accel_y', 'accel_z', 'speed')
INDICES = (0, 1, 2, 6)


def select_channels(x, mask):
    """Project canonical [...,7] records to [...,4], preserving missing speed."""
    if x.shape[-1] != 7 or mask.shape != x.shape or mask.dtype != torch.bool:
        raise ValueError('Expected canonical seven-channel data and a matching boolean mask')
    return x[..., list(INDICES)], mask[..., list(INDICES)]


class AccelerationSpeedDataset(RoadDataset):
    """Same files, windows and labels as RoadDataset; x/mask are [time,4].

    No global normalization statistics are exposed. Use InstancePatchTST.
    Projection happens once per cached recording, so indexing helpers also
    see four-channel validity. Gyro-only samples become unobserved correctly.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_stats = None

    def _open(self, record_index):
        arrays = super()._open(record_index)
        if arrays['x'].shape[-1] == 7:
            # Advanced indexing creates in-memory copies, never edits the files.
            arrays['x'] = arrays['x'][:, list(INDICES)]
            arrays['mask'] = arrays['mask'][:, list(INDICES)]
        return arrays


def augment(x, mask, *, seed, step, rotate=True, noise=True,
            yaw_degrees=20., tilt_degrees=10., probability=.75):
    """Apply the existing mounting/noise recipe without reading real gyro.

    Padding solely adapts the seven-channel augmentation interface. Its gyro
    triad is always absent and discarded; the model still has four channels.
    """
    if x.ndim != 3 or x.shape[-1] != 4 or mask.shape != x.shape or mask.dtype != torch.bool:
        raise ValueError('Expected [batch,time,4] data and a matching boolean mask')
    padded = x.new_zeros(*x.shape[:-1], 7)
    observed = torch.zeros_like(padded, dtype=torch.bool)
    padded[..., list(INDICES)] = x
    observed[..., list(INDICES)] = mask
    result = imu_augmentation(padded, observed, seed=seed, step=step, rotate=rotate, noise=noise,
                              yaw_degrees=yaw_degrees, tilt_degrees=tilt_degrees, probability=probability)
    return result[..., list(INDICES)]


class CanonicalInputAdapter(nn.Module):
    """Use a four-input model with existing seven-channel evaluation loaders."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.channels = 7

    def forward(self, x, mask):
        x, mask = select_channels(x, mask)
        return self.model(x, mask)
