"""Masked per-window normalization for the bidirectional road model.

No dataset-fitted normalization or running statistics are used. Statistics
come only from observed values inside the current input window. This is a
supervised bidirectional encoder, not a causal/hidden-patch pretrainer.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from imu_inference.patchtst import PatchTST, PatchTSTRoadModel


def instance_normalize(x, mask=None, eps=1e-5):
    """Normalize each [window, channel] over observed time samples in FP32.

    Return normalized values, validity and [B,C,3] descriptors: asinh(mean),
    log(standard deviation), observed fraction. Missing channels stay zero.
    The optional descriptors preserve measured scale; they are not a second
    normalization fitted across recordings or evaluation examples.
    """
    if x.ndim != 3 or not x.is_floating_point() or eps <= 0 or not math.isfinite(eps):
        raise ValueError("Expected floating [B,T,C] input and positive epsilon")
    if mask is None:
        mask = torch.isfinite(x)
    if mask.shape != x.shape or mask.dtype != torch.bool:
        raise ValueError("Boolean mask must match inputs")
    if not x.shape[0] or not x.shape[1] or not torch.isfinite(x[mask]).all():
        raise ValueError("Observed inputs must be finite and nonempty")
    safe = x.float().masked_fill(~mask, 0)
    count = mask.sum(1, keepdim=True)
    mean = safe.sum(1, keepdim=True)/count.clamp_min(1)
    centered = (safe-mean).masked_fill(~mask, 0)
    variance = centered.square().sum(1, keepdim=True)/count.clamp_min(1)
    scale = (variance+eps).sqrt()
    normalized = (centered/scale).masked_fill(~mask, 0)
    descriptors = torch.stack((mean.squeeze(1).asinh(), scale.squeeze(1).log(),
                               count.squeeze(1)/x.shape[1]), dim=-1)
    descriptors = descriptors.masked_fill((count.squeeze(1)==0)[..., None], 0)
    return normalized, mask, descriptors


class InstancePatchTST(PatchTST):
    """Existing patch encoder with per-instance, per-channel standardization."""
    def __init__(self, *, instance_eps=1e-5, **kwargs):
        if kwargs.pop("train_stats", None) is not None:
            raise ValueError("InstancePatchTST does not accept global statistics")
        super().__init__(**kwargs)
        self.instance_eps = instance_eps
        self.config["instance_eps"] = instance_eps

    def patchify(self, x, valid_mask=None):
        normalized, mask, _ = instance_normalize(x, valid_mask, self.instance_eps)
        return self._patches(normalized, mask)

    def _patches(self, x, mask):
        if x.shape[-1] != self.channels or x.shape[1] % self.patch_length:
            raise ValueError("Expected configured channels and whole patches")
        if x.shape[1] // self.patch_length > self.max_patches:
            raise ValueError("Input exceeds configured patch count")
        return (x.transpose(1,2).unfold(-1,self.patch_length,self.patch_length),
                mask.transpose(1,2).unfold(-1,self.patch_length,self.patch_length))

    def forward(self, x, valid_mask=None, patch_mask=None):
        if patch_mask is not None:
            raise ValueError("Hidden-patch pretraining needs visible-only statistics")
        normalized, mask, stats = instance_normalize(x, valid_mask, self.instance_eps)
        patches, observed = self._patches(normalized, mask)
        features, valid = self.encode_patches(patches, observed)
        return dict(features=features, patch_valid=valid, statistics=stats)


class InstanceRoadModel(PatchTSTRoadModel):
    """Two patch heads; optional window-scale descriptors preserve amplitude."""
    def __init__(self, encoder, *, statistics_mode="both", dropout=.1):
        if statistics_mode not in ("none", "roughness", "both"):
            raise ValueError("statistics_mode must be none, roughness or both")
        super().__init__(encoder, dropout)
        self.statistics_mode = statistics_mode
        self.statistics_embedding = (None if statistics_mode == "none" else
                                     nn.Sequential(nn.Linear(3, encoder.d_model), nn.GELU()))

    def forward(self, x, valid_mask=None):
        encoded = self.encoder(x, valid_mask)
        features, valid = encoded["features"], encoded["patch_valid"]
        b, c, n, d = features.shape
        enriched = features
        if self.statistics_embedding is not None:
            extra = self.statistics_embedding(encoded["statistics"])[:, :, None, :]
            enriched = (features+extra).masked_fill(~valid[..., None], 0)
        flatten = lambda z: z.permute(0,2,1,3).reshape(b,n,c*d)
        disturbance = enriched if self.statistics_mode == "both" else features
        return dict(roughness=F.softplus(self.roughness_head(flatten(enriched)).squeeze(-1).float()),
                    disturbance_logit=self.disturbance_head(flatten(disturbance)).squeeze(-1),
                    patch_valid=valid.any(1))
