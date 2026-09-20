"""Small sensor-frame perturbations; no time warps or target changes."""
import math
import torch
from torch import nn


def rotations(batch, generator, device, maximum_degrees=5.):
    """One proper, bounded axis-angle rotation per input window."""
    axis = torch.randn(batch, 3, generator=generator, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    angle = (2 * torch.rand(batch, generator=generator, device=device) - 1) * math.radians(maximum_degrees)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(batch, 3, 3)
    eye = torch.eye(3, device=device).expand(batch, 3, 3)
    return eye + angle.sin()[:, None, None] * skew + (1-angle.cos())[:, None, None] * (skew @ skew)


def perturb(x, mask, *, seed, step, rotate=False, noise=False):
    """Raw SI inputs in/out. Preserve masks, missing zeros and sample alignment.

    Rotation is shared by acceleration (including gravity) and gyro. A vector
    containing a missing component is left unchanged. Noise is tied across
    identical held readings, so it does not invent a higher acquisition rate.
    Conservative amplitudes are fixed priors, not fitted to validation results.
    """
    if x.ndim != 3 or x.shape[-1] != 7 or mask.shape != x.shape or mask.dtype != torch.bool:
        raise ValueError('Expected [B,T,7] values and boolean masks')
    result = x.clone()
    if rotate:
        generator = torch.Generator(device=x.device).manual_seed(seed * 1000003 + step * 7 + 301)
        matrix = rotations(len(x), generator, x.device)
        for offset in (0, 3):
            vector = x[..., offset:offset+3]
            valid = mask[..., offset:offset+3].all(-1, keepdim=True)
            result[..., offset:offset+3] = torch.where(valid, vector @ matrix.transpose(-1, -2), vector)
    if noise:
        generator = torch.Generator(device=x.device).manual_seed(seed * 1000003 + step * 7 + 302)
        for offset, jitter, bias in ((0, .005, .02), (3, .0002, .001)):
            original = x[..., offset:offset+3]
            valid = mask[..., offset:offset+3]
            changed = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            changed[:, 1:] = ((original[:, 1:] != original[:, :-1]).any(-1)
                              | (valid[:, 1:] != valid[:, :-1]).any(-1))
            positions = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
            source = torch.where(changed, positions, 0).cummax(1).values
            samples = torch.randn(original.shape, generator=generator, device=x.device).clamp(-3, 3) * jitter
            samples = samples.gather(1, source[..., None].expand(-1, -1, 3))
            offset_bias = (2 * torch.rand((len(x), 1, 3), generator=generator, device=x.device)-1) * bias
            result[..., offset:offset+3] += samples + offset_bias
    return result.masked_fill(~mask, 0)


class AugmentedModel(nn.Module):
    """Use existing losses/targets with augmentation active only during training."""
    def __init__(self, model, seed, rotate=False, noise=False):
        super().__init__()
        self.model, self.seed = model, seed
        self.rotate, self.noise, self.steps = rotate, noise, 0

    @property
    def encoder(self):
        return self.model.encoder

    def forward(self, x, mask):
        if self.training:
            if self.rotate or self.noise:
                x = perturb(x, mask, seed=self.seed, step=self.steps, rotate=self.rotate, noise=self.noise)
            self.steps += 1
        return self.model(x, mask)
