"""Fixed phone-mount errors, applied to raw IMU before normalization.

Yaw is about the window's approximate vertical, inferred from its observed
mean total acceleration. Tilt is about a random horizontal axis. This handles
both Z-up Kaggle/LiRA and Y-up RoadSens without pretending to calibrate them
into a common vehicle frame. Use only on TRAIN, not in live preprocessing.
"""
import math

import torch
from torch.nn import functional as F

from road_training.augmentation import perturb as sensor_noise


def axis_angle(axis, angle):
    """Proper rotation matrices for unit axes [B,3] and angles [B] in radians."""
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(len(axis), -1, -1)
    return eye + angle.sin()[:, None, None] * skew + (1-angle.cos())[:, None, None] * (skew @ skew)


def mounting_rotations(x, mask, *, seed, step, yaw_degrees=20., tilt_degrees=10., probability=.75):
    """Return one FP32 rotation per window, shared by accel and gyro.

    Nonzero angles are uniform in their signed bounds. A fraction 1-probability
    keeps its original orientation. Mean acceleration is only an approximate
    up direction; gravity must be included. No labels or other windows enter.

    With a partial XYZ vector, rotation would require an unobserved component.
    Leave that entire window's orientation unchanged instead of inventing data
    or switching coordinate systems partway through a window. Entirely absent
    triads (LiRA gyro), padding, and missing speed are supported normally.
    """
    if (x.ndim != 3 or x.shape[-1] != 7 or not x.is_floating_point()
            or mask.shape != x.shape or mask.dtype != torch.bool or mask.device != x.device
            or not x.shape[0] or not x.shape[1]):
        raise ValueError('Expected nonempty floating [B,T,7] input and matching boolean mask')
    for value, limit, name in ((yaw_degrees, 180., 'yaw_degrees'),
                               (tilt_degrees, 90., 'tilt_degrees'), (probability, 1., 'probability')):
        if not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError(f'{name} must be finite and between 0 and {limit}')
    if not torch.isfinite(x[mask]).all():
        raise ValueError('Observed inputs must be finite')
    # Explicit FP32 prevents outer BF16 autocast from distorting the rotation.
    with torch.autocast(device_type=x.device.type, enabled=False):
        accel_valid = mask[..., :3].all(-1)
        accel = x[..., :3].detach().float().masked_fill(~accel_valid[..., None], 0)
        mean = accel.sum(1) / accel_valid.sum(1).clamp_min(1)[:, None]
        magnitude = mean.norm(dim=-1)
        eligible = (magnitude >= 5.) & (magnitude <= 15.)
        for offset in (0, 3):
            observed = mask[..., offset:offset+3]
            partial = observed.any(-1) & ~observed.all(-1)
            eligible &= ~partial.any(-1)
        up = mean / magnitude.clamp_min(1e-6)[:, None]
        fallback = torch.tensor([0., 0., 1.], device=x.device).expand_as(up)
        up = torch.where(eligible[:, None], up, fallback)
        # Pick the least parallel coordinate axis for a stable horizontal basis.
        reference = F.one_hot(up.abs().argmin(-1), num_classes=3).float()
        horizontal = F.normalize(torch.linalg.cross(up, reference), dim=-1)
        other = torch.linalg.cross(up, horizontal)
        generator = torch.Generator(device=x.device).manual_seed(seed * 1000003 + step * 7 + 401)
        draws = torch.rand(len(x), 4, device=x.device, generator=generator)
        yaw = (2*draws[:, 0]-1) * math.radians(yaw_degrees)
        tilt = (2*draws[:, 1]-1) * math.radians(tilt_degrees)
        direction = draws[:, 2] * (2*math.pi)
        tilt_axis = direction.cos()[:, None]*horizontal + direction.sin()[:, None]*other
        matrix = axis_angle(tilt_axis, tilt) @ axis_angle(up, yaw)
        active = eligible & (draws[:, 3] < probability)
        eye = torch.eye(3, device=x.device).expand_as(matrix)
        return torch.where(active[:, None, None], matrix, eye)


def perturb(x, mask, *, seed, step, rotate=True, noise=True,
            yaw_degrees=20., tilt_degrees=10., probability=.75):
    """Training augmentation compatible with instance_study's perturb call.

    Rotation preserves vector magnitude, speed, masks, timestamps and targets.
    Existing small sensor noise/bias is added separately when noise=True.
    """
    matrix = mounting_rotations(x, mask, seed=seed, step=step,
        yaw_degrees=yaw_degrees, tilt_degrees=tilt_degrees,
        probability=probability if rotate else 0.)
    with torch.autocast(device_type=x.device.type, enabled=False):
        result = x.masked_fill(~mask, 0).clone()
        for offset in (0, 3):
            result[..., offset:offset+3] = (
                result[..., offset:offset+3].float() @ matrix.transpose(-1, -2)).to(x.dtype)
    return sensor_noise(result, mask, seed=seed, step=step, rotate=False, noise=noise)
