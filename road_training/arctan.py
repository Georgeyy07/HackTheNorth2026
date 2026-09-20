"""Patch reconstruction from the supplied ArcTan Diffusion paper, Sec. 3.

An x-predicting encoder/decoder DiT, trained with ArcTanLoss in velocity space.
The encoder (including its projection) transfers at t=1; discard the decoder.
Road adaptation: fixed TRAIN z-scores and explicit missing-value masks.
"""
import math

import torch
from torch import nn

from road_training.patchtst import normalized_patches


def arctan_loss(target_velocity, predicted_velocity):
    """Elementwise paper Eq. (4), including gamma * predicted_velocity**2.

    Keep this arithmetic FP32 under autocast. The final term penalizes the
    prediction, NOT the residual; perfect reconstruction need not have zero
    total loss. Constants are the paper's fixed published defaults.
    """
    prediction = predicted_velocity.float()
    error = target_velocity.float() - prediction
    a, k, alpha, beta, gamma = 1.1, 1.3, .5, .5, .05
    smooth = a * error * torch.atan(k * error) - a / (2 * k) * torch.log1p((k * error).square())
    return alpha * smooth + beta * error.abs() + gamma * prediction.square()


class TimestepEmbedding(nn.Module):
    """Continuous sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, d_model, frequency_dim=256):
        super().__init__()
        frequencies = torch.exp(-math.log(10000) * torch.arange(frequency_dim // 2) / (frequency_dim // 2))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(frequency_dim, d_model), nn.SiLU(),
                                 nn.Linear(d_model, d_model))

    def forward(self, t):
        angles = t.float().unsqueeze(-1) * self.frequencies
        return self.mlp(torch.cat((angles.cos(), angles.sin()), dim=-1))


class DiTBlock(nn.Module):
    """Pre-norm attention/FFN with per-token AdaLN-Zero conditioning."""

    def __init__(self, d_model, n_heads, ffn_dim, dropout):
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm_ffn = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, d_model), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, condition, valid):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(condition).chunk(6, dim=-1)
        z = self.norm_attention(x) * (1 + scale_a) + shift_a
        z = self.attention(z, z, z, key_padding_mask=~valid, need_weights=False)[0]
        x = x + gate_a * self.dropout(z)
        z = self.norm_ffn(x) * (1 + scale_f) + shift_f
        x = x + gate_f * self.ffn(z)
        return x.masked_fill(~valid.unsqueeze(-1), 0)


class ArcTanEncoder(nn.Module):
    """DiT encoder with the same downstream interface as PatchTST.

    channel_mode='mixed': a token contains C*P values (paper classification).
    channel_mode='independent': a token contains P values (paper forecasting).
    Features are [B,F,N,D], where F=1 for mixed and F=C for independent.
    forward() always extracts clean t=1 features, without diffusion sampling.
    """

    def __init__(self, channels=7, patch_length=16, d_model=128, n_heads=16,
                 n_layers=2, ffn_dim=512, dropout=.1, max_patches=256,
                 channel_mode="mixed", train_stats=None):
        super().__init__()
        sizes = (channels, patch_length, d_model, n_heads, n_layers, ffn_dim, max_patches)
        if any(type(v) is not int or v < 1 for v in sizes):
            raise ValueError("Model dimensions must be positive integers")
        if d_model % n_heads or not 0 <= dropout < 1:
            raise ValueError("Invalid number of heads or dropout")
        if channel_mode not in ("mixed", "independent"):
            raise ValueError("channel_mode must be 'mixed' or 'independent'")
        self.config = dict(channels=channels, patch_length=patch_length, d_model=d_model,
                           n_heads=n_heads, n_layers=n_layers, ffn_dim=ffn_dim,
                           dropout=dropout, max_patches=max_patches, channel_mode=channel_mode)
        self.channels, self.patch_length, self.d_model = channels, patch_length, d_model
        self.max_patches, self.channel_mode = max_patches, channel_mode
        self.feature_channels = 1 if channel_mode == "mixed" else channels
        self.token_width = patch_length * (channels if channel_mode == "mixed" else 1)
        mean = torch.zeros(channels) if train_stats is None else torch.tensor(train_stats["mean"], dtype=torch.float32)
        std = torch.ones(channels) if train_stats is None else torch.tensor(train_stats["std"], dtype=torch.float32)
        if mean.shape != (channels,) or std.shape != mean.shape:
            raise ValueError("One TRAIN mean/std is required per channel")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Normalization must be finite with positive std")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.value_embedding = nn.Linear(self.token_width, d_model)
        self.valid_embedding = nn.Linear(self.token_width, d_model, bias=False)
        self.position = nn.Parameter(torch.empty(1, max_patches, d_model))
        nn.init.normal_(self.position, std=.02)
        self.time_embedding = TimestepEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DiTBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)])
        # Paper Eq. (2): this projection belongs to the transferred encoder.
        self.projection = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

    def patchify(self, x, valid_mask=None):
        return normalized_patches(x, valid_mask, self.mean, self.std,
                                  self.patch_length, self.max_patches)

    def pack(self, patches):
        """[B,C,N,P] -> [B*F,N,token_width], channel-major within each token."""
        b, c, n, p = patches.shape
        if self.channel_mode == "mixed":
            return patches.permute(0, 2, 1, 3).reshape(b, n, c * p)
        return patches.reshape(b * c, n, p)

    def unpack(self, tokens, batch_size):
        """Inverse of pack(), restoring scalar predictions to [B,C,N,P]."""
        n = tokens.shape[1]
        if self.channel_mode == "mixed":
            return tokens.reshape(batch_size, n, self.channels, self.patch_length).permute(0, 2, 1, 3)
        return tokens.reshape(batch_size, self.channels, n, self.patch_length)

    def encode_patches(self, patches, observed, timesteps):
        b, _, n, _ = patches.shape
        values, observed_tokens = self.pack(patches), self.pack(observed)
        valid = observed_tokens.any(-1)
        tokens = self.value_embedding(values.masked_fill(~observed_tokens, 0))
        tokens = tokens + self.valid_embedding(observed_tokens.to(values.dtype))
        tokens = self.dropout(tokens + self.position[:, :n].to(tokens.dtype))
        condition = self.time_embedding(timesteps.reshape(b * self.feature_channels, n))
        # Skip entirely missing sequences, so attention never sees all keys masked.
        present = valid.any(-1)
        features = torch.zeros_like(tokens)
        if present.any():
            z = tokens[present]
            for layer in self.layers:
                z = layer(z, condition[present], valid[present])
            features[present] = self.projection(z).to(features.dtype)
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        return features, valid, condition

    def forward(self, x, valid_mask=None):
        patches, observed = self.patchify(x, valid_mask)
        b, _, n, _ = patches.shape
        t = torch.ones(b, self.feature_channels, n, device=x.device)
        features, valid, _ = self.encode_patches(patches, observed, t)
        return dict(features=features.reshape(b, self.feature_channels, n, self.d_model),
                    patch_valid=valid.reshape(b, self.feature_channels, n))


class ArcTanPretrainer(nn.Module):
    """Predict clean patches; score velocity residuals with the exact ArcTan loss.

    z = t*x + (1-t)*noise. Both velocities use max(1-t, .05), including
    the TARGET velocity near t=1. Loss covers all originally observed patch
    components. This is denoising reconstruction, not a masked-patch MAE.
    """

    def __init__(self, encoder, decoder_layers=2, timestep_mode="token", tau=.05):
        super().__init__()
        if type(decoder_layers) is not int or decoder_layers < 1:
            raise ValueError("decoder_layers must be a positive integer")
        if timestep_mode not in ("token", "sequence") or not 0 < tau <= 1:
            raise ValueError("Invalid timestep_mode or velocity denominator floor")
        self.encoder, self.timestep_mode, self.tau = encoder, timestep_mode, tau
        self.config = dict(decoder_layers=decoder_layers, timestep_mode=timestep_mode, tau=tau)
        config = encoder.config
        self.decoder = nn.ModuleList([
            DiTBlock(encoder.d_model, config["n_heads"], config["ffn_dim"], config["dropout"])
            for _ in range(decoder_layers)
        ])
        self.output_norm = nn.LayerNorm(encoder.d_model, elementwise_affine=False, eps=1e-6)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(encoder.d_model, 2 * encoder.d_model))
        self.head = nn.Linear(encoder.d_model, encoder.token_width)
        # Standard DiT initialization: zero modulation and output projection.
        for module in (self.output_modulation[-1], self.head):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, valid_mask=None, *, generator=None, timesteps=None, noise=None):
        patches, observed = self.encoder.patchify(x, valid_mask)
        target = patches.detach()
        b, c, n, _ = patches.shape
        shape = (b, self.encoder.feature_channels, n)
        if timesteps is None:
            sample_shape = shape if self.timestep_mode == "token" else (b, 1, 1)
            timesteps = torch.rand(sample_shape, device=x.device, generator=generator).expand(shape)
        if timesteps.shape != shape or not torch.isfinite(timesteps).all():
            raise ValueError(f"timesteps must be finite with shape {shape}")
        if (timesteps < 0).any() or (timesteps > 1).any():
            raise ValueError("timesteps must be in [0, 1]")
        if noise is None:
            noise = torch.randn(patches.shape, device=x.device, generator=generator)
        if noise.shape != patches.shape or not torch.isfinite(noise[observed]).all():
            raise ValueError("noise must match [B,C,N,P] and be finite where observed")
        noise = noise.detach().float().masked_fill(~observed, 0)
        t = timesteps.detach().float().expand(b, c, n).unsqueeze(-1)
        noisy = (t * patches + (1 - t) * noise).masked_fill(~observed, 0)
        features, valid, condition = self.encoder.encode_patches(noisy, observed, timesteps)
        present = valid.any(-1)
        decoded = torch.zeros_like(features)
        if present.any():
            z = features[present]
            for layer in self.decoder:
                z = layer(z, condition[present], valid[present])
            shift, scale = self.output_modulation(condition[present]).chunk(2, dim=-1)
            decoded[present] = (self.output_norm(z) * (1 + scale) + shift).to(decoded.dtype)
        prediction = self.encoder.unpack(self.head(decoded), b)
        denominator = (1 - t).clamp(min=self.tau)
        target_velocity = (target.float() - noisy.detach().float()) / denominator
        predicted_velocity = (prediction.float() - noisy.float()) / denominator
        error = arctan_loss(target_velocity[observed], predicted_velocity[observed])
        loss = error.mean() if error.numel() else prediction.float().sum() * 0
        return dict(loss=loss, prediction=prediction, target=target, loss_mask=observed,
                    noisy=noisy, timesteps=timesteps, target_velocity=target_velocity,
                    predicted_velocity=predicted_velocity)
