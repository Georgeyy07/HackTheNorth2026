"""DropPatch: discard patches, then reconstruct masked retained patches.

Paper: https://arxiv.org/abs/2412.15315 (AAAI 2025).
This adapts its drop/mask objective to our existing, channel-independent
PatchTST encoder, TRAIN normalization and sensor-availability masks.
"""
import torch
from torch import nn


def gather_patches(values, indices):
    """Gather [B,C,N,D] at [B,C,K] indices, keeping the last dimension."""
    return values.gather(2, indices.unsqueeze(-1).expand(*indices.shape, values.shape[-1]))


class DropPatchPretrainer(nn.Module):
    """Default: drop 60%, then mask 40% of the retained patches.

    Dropped patches enter neither attention nor loss. Masked retained patches
    enter attention as zero-valued patches with their ORIGINAL positions.
    Predictions/targets are [B,C,K,P]; kept_indices maps K back to full N.
    Only observed values in masked retained patches contribute to MSE.
    """

    def __init__(self, encoder, drop_ratio=.6, mask_ratio=.4):
        super().__init__()
        if not 0 <= drop_ratio < 1 or not 0 < mask_ratio < 1:
            raise ValueError("Require 0 <= drop_ratio < 1 and 0 < mask_ratio < 1")
        self.encoder = encoder
        self.drop_ratio, self.mask_ratio = drop_ratio, mask_ratio
        self.config = dict(drop_ratio=drop_ratio, mask_ratio=mask_ratio)
        self.head = nn.Linear(encoder.d_model, encoder.patch_length)

    def sample_kept(self, observed, generator=None):
        """Sample independently per channel; pad shorter observed sequences.

        With missing data, apply the drop ratio to available patches. Keep at
        least two when possible (one context, one target). Padding never enters
        attention or loss; it merely lets different counts share one batch.
        """
        valid = observed.any(-1)
        n = valid.shape[-1]
        k = min(n, max(2, int(n * (1 - self.drop_ratio))))
        counts = valid.sum(-1)
        retain = (counts * (1 - self.drop_ratio)).floor().long().clamp(min=2)
        retain = torch.minimum(retain, counts)
        scores = torch.rand(valid.shape, device=valid.device, generator=generator)
        indices = scores.masked_fill(~valid, 2).argsort(-1)[..., :k]
        slots = torch.arange(k, device=valid.device) < retain.unsqueeze(-1)
        return indices, slots

    def forward(self, x, valid_mask=None, *, generator=None,
                kept_indices=None, patch_mask=None):
        patches, observed = self.encoder.patchify(x, valid_mask)
        if patches.shape[2] < 2:
            raise ValueError("DropPatch needs at least two patches per window")
        if kept_indices is None:
            kept_indices, slots = self.sample_kept(observed, generator)
        else:
            if (kept_indices.ndim != 3 or kept_indices.shape[:2] != patches.shape[:2]
                    or kept_indices.dtype != torch.long or kept_indices.shape[-1] < 1):
                raise ValueError("kept_indices must be int64 [batch, channels, retained]")
            if (kept_indices < 0).any() or (kept_indices >= patches.shape[2]).any():
                raise ValueError("kept_indices are outside the input window")
            ordered = kept_indices.sort(-1).values
            if (ordered[..., 1:] == ordered[..., :-1]).any():
                raise ValueError("kept_indices must not contain duplicate patches")
            slots = torch.ones_like(kept_indices, dtype=torch.bool)
        retained = gather_patches(patches, kept_indices)
        observed = gather_patches(observed, kept_indices) & slots.unsqueeze(-1)
        valid = observed.any(-1)
        if patch_mask is None:
            counts = valid.sum(-1)
            # As in the paper's code: visible = floor(K * (1 - mask_ratio)).
            visible = (counts * (1 - self.mask_ratio)).floor().long().clamp(min=1)
            hidden = (counts - visible).clamp(min=0)
            scores = torch.rand(valid.shape, device=x.device, generator=generator)
            ranks = scores.masked_fill(~valid, 2).argsort(-1).argsort(-1)
            patch_mask = (ranks < hidden.unsqueeze(-1)) & valid
        if patch_mask.shape != valid.shape or patch_mask.dtype != torch.bool:
            raise ValueError("patch_mask must be boolean [batch, channels, retained]")
        if (patch_mask & ~valid).any():
            raise ValueError("Cannot reconstruct an unavailable retained patch")
        features, _ = self.encoder.encode_patches(
            retained, observed, patch_mask, position_ids=kept_indices, mask_style="zero")
        prediction = self.head(features)
        target = retained.detach()
        loss_mask = observed & patch_mask.unsqueeze(-1)
        error = prediction.float()[loss_mask] - target.float()[loss_mask]
        loss = error.square().mean() if error.numel() else prediction.float().sum() * 0
        return dict(loss=loss, prediction=prediction, target=target,
                    loss_mask=loss_mask, patch_mask=patch_mask,
                    kept_indices=kept_indices, retained_mask=valid)
