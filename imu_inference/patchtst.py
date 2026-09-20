"""PatchTST-style bidirectional encoder, masked pretrainer and downstream head.

Each variate is patched separately; embedding and Transformer weights are
shared across variates. Only the downstream head combines sensor channels.
Reference: https://arxiv.org/abs/2211.14730 (an adaptation, not a reproduction).
"""
import torch
from torch import nn
from torch.nn import functional as F


def normalized_patches(x, valid_mask, mean, std, patch_length, max_patches):
    """Fixed TRAIN normalization, then non-overlapping [B,C,N,P] patches.

    Shared by PatchTST and the diffusion encoder. Invalid payloads never enter
    arithmetic; normalization cannot reveal hidden targets through window stats.
    """
    if x.ndim != 3 or x.shape[-1] != mean.numel() or not x.is_floating_point():
        raise ValueError("x must be floating point [batch, time, channels]")
    if x.shape[0] == 0 or x.shape[1] == 0 or x.shape[1] % patch_length:
        raise ValueError("Use a nonempty batch and time length divisible by patch_length")
    if x.shape[1] // patch_length > max_patches:
        raise ValueError("Window exceeds max_patches")
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    if valid_mask.shape != x.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with the same shape as x")
    if not torch.isfinite(x[valid_mask]).all():
        raise ValueError("Observed input values must be finite")
    x = x.masked_fill(~valid_mask, 0)
    x = ((x - mean) / std).masked_fill(~valid_mask, 0)
    return (x.transpose(1, 2).unfold(-1, patch_length, patch_length),
            valid_mask.transpose(1, 2).unfold(-1, patch_length, patch_length))


class PatchTST(nn.Module):
    """[batch, time, channels] -> [batch, channels, patches, d_model].

    Supply dataset.train_stats to accept raw SI inputs with fixed training-set
    z-score normalization. If omitted, normalization is the identity. There is
    no per-window normalization or fitting during forward(). Patches do not
    overlap, and window length must be divisible by patch_length.
    """

    def __init__(self, channels=7, patch_length=16, d_model=128, n_heads=4,
                 n_layers=3, ffn_dim=512, dropout=0.1, max_patches=256,
                 train_stats=None):
        super().__init__()
        sizes = [channels, patch_length, d_model, n_heads, n_layers, ffn_dim, max_patches]
        if any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("Model dimensions must be positive integers")
        if d_model % n_heads or not 0 <= dropout < 1:
            raise ValueError("d_model must be divisible by n_heads; dropout must be in [0, 1)")
        self.config = dict(channels=channels, patch_length=patch_length, d_model=d_model,
                           n_heads=n_heads, n_layers=n_layers, ffn_dim=ffn_dim,
                           dropout=dropout, max_patches=max_patches)
        self.channels, self.patch_length, self.d_model = channels, patch_length, d_model
        self.max_patches = max_patches
        mean = torch.zeros(channels) if train_stats is None else torch.tensor(train_stats["mean"], dtype=torch.float32)
        std = torch.ones(channels) if train_stats is None else torch.tensor(train_stats["std"], dtype=torch.float32)
        if mean.shape != (channels,) or std.shape != mean.shape:
            raise ValueError("One training mean/std is required per channel")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Normalization statistics must be finite; std must be positive")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.value_embedding = nn.Linear(patch_length, d_model)
        self.valid_embedding = nn.Linear(patch_length, d_model, bias=False)
        self.position = nn.Parameter(torch.empty(1, 1, max_patches, d_model))
        self.mask_token = nn.Parameter(torch.empty(1, 1, 1, d_model))
        nn.init.normal_(self.position, std=.02)
        nn.init.normal_(self.mask_token, std=.02)
        self.dropout = nn.Dropout(dropout)
        # Construct each layer separately so initialization is independent.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def patchify(self, x, valid_mask=None):
        """Return normalized patches and observed-value masks, both [B,C,N,P]."""
        return normalized_patches(x, valid_mask, self.mean, self.std,
                                  self.patch_length, self.max_patches)

    def encode_patches(self, patches, observed, patch_mask=None, *,
                       position_ids=None, mask_style="token"):
        """Encode patches, optionally gathered at their original position_ids.

        mask_style='zero' embeds zero-valued hidden patches for DropPatch;
        the original masked pretrainer retains its learned 'token' behavior.
        """
        valid = observed.any(-1)  # [B,C,N]; missing patches cannot be keys.
        if patch_mask is None:
            patch_mask = torch.zeros_like(valid)
        if patch_mask.shape != valid.shape or patch_mask.dtype != torch.bool:
            raise ValueError("patch_mask must be boolean [batch, channels, patches]")
        visible = observed & ~patch_mask.unsqueeze(-1)
        tokens = self.value_embedding(patches.masked_fill(~visible, 0))
        tokens = tokens + self.valid_embedding(visible.to(patches.dtype))
        # Preserve autocast's token dtype instead of promoting every activation
        # back to FP32 when adding the learned mask/position parameters.
        if mask_style == "token":
            tokens = torch.where(patch_mask.unsqueeze(-1), self.mask_token.to(tokens.dtype), tokens)
        elif mask_style != "zero":
            raise ValueError("mask_style must be 'token' or 'zero'")
        if position_ids is None:
            position = self.position[:, :, :tokens.shape[2]]
        else:
            if position_ids.shape != valid.shape or position_ids.dtype != torch.long:
                raise ValueError("position_ids must be int64 [batch, channels, patches]")
            if (position_ids < 0).any() or (position_ids >= self.max_patches).any():
                raise ValueError("position_ids are outside the position table")
            position = self.position[0, 0][position_ids]
        tokens = self.dropout(tokens + position.to(tokens.dtype))
        b, c, n, d = tokens.shape
        flat, key_valid = tokens.reshape(b*c, n, d), valid.reshape(b*c, n)
        # Skip fully missing channel sequences: attention with every key
        # masked can produce NaNs. Returned missing features stay zero.
        present = key_valid.any(-1)
        encoded = torch.zeros_like(flat)
        if present.any():
            z = flat[present]
            for layer in self.layers:
                z = layer(z, src_key_padding_mask=~key_valid[present])
            # Autocast can run LayerNorm in FP32; indexed writes require the
            # destination dtype, and stored features should remain compact.
            encoded[present] = self.norm(z).to(encoded.dtype)
        encoded = encoded.masked_fill(~key_valid.unsqueeze(-1), 0)
        return encoded.reshape(b, c, n, d), valid

    def forward(self, x, valid_mask=None, patch_mask=None):
        patches, observed = self.patchify(x, valid_mask)
        features, patch_valid = self.encode_patches(patches, observed, patch_mask)
        return {"features": features, "patch_valid": patch_valid}


class PatchTSTPretrainer(nn.Module):
    """Reconstruct hidden patches using bidirectional context and masked L1/MSE.

    Predictions and targets use normalized units and shape [B,C,N,P]. Loss
    includes only originally observed values inside hidden patches. Random
    masks leave at least one visible patch per channel when possible.
    """

    def __init__(self, encoder, loss="l1"):
        super().__init__()
        if loss not in ("l1", "mse"):
            raise ValueError("loss must be 'l1' or 'mse'")
        self.encoder, self.loss_type = encoder, loss
        self.head = nn.Linear(encoder.d_model, encoder.patch_length)

    @staticmethod
    def random_mask(patch_valid, mask_ratio):
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be between zero and one")
        count = patch_valid.sum(-1)
        hide = (count * mask_ratio).round().long().clamp(min=1)
        hide = torch.minimum(hide, (count-1).clamp(min=0))
        scores = torch.rand(patch_valid.shape, device=patch_valid.device).masked_fill(~patch_valid, 2)
        rank = scores.argsort(-1).argsort(-1)
        return (rank < hide.unsqueeze(-1)) & patch_valid

    def forward(self, x, valid_mask=None, mask_ratio=.4, patch_mask=None):
        patches, observed = self.encoder.patchify(x, valid_mask)
        if patch_mask is None:
            patch_mask = self.random_mask(observed.any(-1), mask_ratio)
        features, _ = self.encoder.encode_patches(patches, observed, patch_mask)
        prediction = self.head(features)
        target = patches.detach()
        loss_mask = observed & patch_mask.unsqueeze(-1)
        error = prediction[loss_mask] - target[loss_mask]
        if error.numel():
            loss = error.abs().mean() if self.loss_type == "l1" else error.square().mean()
        else:
            loss = prediction.sum() * 0  # Differentiable zero; caller can skip.
        return {"loss": loss, "prediction": prediction, "target": target,
                "loss_mask": loss_mask, "patch_mask": patch_mask}


class PatchTSTPredictor(nn.Module):
    """Combine channel embeddings for classification or regression.

    output_level='patch' (default): output [B,N,output_dim], one prediction
    per patch; suitable for majority-voted labels from the supervised trainer.
    output_level='sample': output [B,T,output_dim], suitable for this dataset's
    per-sample labels; each patch predicts its P samples separately.
    output_level='window': output [B,output_dim], for caller-supplied window
    labels. Returns raw logits/values, with no softmax or assumed label mapping.
    """

    def __init__(self, encoder, output_dim, output_level="patch", dropout=.1):
        super().__init__()
        if type(output_dim) is not int or output_dim < 1:
            raise ValueError("output_dim must be a positive integer")
        if output_level not in ("patch", "sample", "window"):
            raise ValueError("output_level must be 'patch', 'sample', or 'window'")
        self.encoder, self.output_dim, self.output_level = encoder, output_dim, output_level
        outputs = output_dim * (encoder.patch_length if output_level == "sample" else 1)
        width = getattr(encoder, "feature_channels", encoder.channels) * encoder.d_model
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, encoder.d_model),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(encoder.d_model, outputs))

    def forward(self, x, valid_mask=None):
        result = self.encoder(x, valid_mask)
        features, valid = result["features"], result["patch_valid"]
        b, c, n, d = features.shape
        if self.output_level == "window":
            weight = valid.unsqueeze(-1).to(features.dtype)
            pooled = (features * weight).sum(2) / weight.sum(2).clamp(min=1)
            return self.head(pooled.reshape(b, c*d))
        features = features.permute(0, 2, 1, 3).reshape(b, n, c*d)
        prediction = self.head(features)
        if self.output_level == "patch":
            return prediction
        return prediction.reshape(b, n*self.encoder.patch_length, self.output_dim)


class PatchTSTRoadModel(nn.Module):
    """One encoder, two independent heads, two predictions at every patch.

    roughness: nonnegative overall section IRI in m/km, [B,N].
    disturbance_logit: binary presence logit, [B,N]; apply sigmoid for probability.
    patch_valid: at least one observed sensor value in the patch, [B,N].
    Both heads use the same bidirectional context; neither pools over time.
    """

    def __init__(self, encoder, dropout=.1):
        super().__init__()
        self.encoder = encoder
        width = getattr(encoder, "feature_channels", encoder.channels) * encoder.d_model

        def head():
            return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, encoder.d_model),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(encoder.d_model, 1))

        self.roughness_head = head()
        self.disturbance_head = head()

    def forward(self, x, valid_mask=None):
        encoded = self.encoder(x, valid_mask)
        features = encoded["features"]
        b, c, n, d = features.shape
        features = features.permute(0, 2, 1, 3).reshape(b, n, c*d)
        return {
            "roughness": F.softplus(self.roughness_head(features).squeeze(-1).float()),
            "disturbance_logit": self.disturbance_head(features).squeeze(-1),
            "patch_valid": encoded["patch_valid"].any(1),
        }
