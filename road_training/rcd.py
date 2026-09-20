"""Time-RCD for road signals: contextual anomaly CE + masked-patch MSE.

Reference: arXiv:2509.21190v5, Sec. 2.2 and Appendix C.4; Apache-2.0
implementation thu-sail-lab/Time-RCD, commit 372bb980426b2f67007311c6f3165ab789c79bef.
This compact adaptation uses fixed TRAIN statistics and explicit missing masks.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from road_training.patchtst import normalized_patches


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
                * self.scale).to(x.dtype)


class RCDAttention(nn.Module):
    """Upstream full-width RoPE and learned same/different-variate attention.

    The per-head bias is b_same for same-channel pairs, b_other otherwise.
    Its row-constant b_other cancels in softmax. Appending channel one-hots
    to Q/K implements the remaining bias exactly and permits fused SDPA.
    Tests compare both outputs and gradients against explicit dense bias.
    """
    def __init__(self, width, heads, channels):
        super().__init__()
        self.width, self.heads, self.channels = width, heads, channels
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.bias = nn.Parameter(torch.zeros(2, heads))
        self.register_buffer('frequencies', 1 / (10000 ** (torch.arange(0, width, 2).float()/width)))

    def rotate(self, x, positions):
        angles = positions.float()[:, None] * self.frequencies
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        pairs = x.reshape(*x.shape[:-1], self.width//2, 2)
        return torch.stack((pairs[..., 0]*cos-pairs[..., 1]*sin,
                            pairs[..., 0]*sin+pairs[..., 1]*cos), -1).flatten(-2)

    def forward(self, x, positions, channel_ids):
        b, length, _ = x.shape
        d = self.width // self.heads
        split = lambda z: z.reshape(b, length, self.heads, d).transpose(1, 2)
        q, k = split(self.rotate(self.q(x), positions)), split(self.rotate(self.k(x), positions))
        v = split(self.v(x))
        onehot = F.one_hot(channel_ids, self.channels).to(q.dtype)[None, None]
        delta = (self.bias[1]-self.bias[0]).to(q.dtype)[None, :, None, None]
        extra = math.ceil(self.channels/8)*8
        q_extra = F.pad(onehot * delta * math.sqrt(d), (0, extra-self.channels)).expand(b, -1, -1, -1)
        k_extra = F.pad(onehot, (0, extra-self.channels)).expand(b, self.heads, -1, -1)
        q, k = torch.cat((q, q_extra), -1), torch.cat((k, k_extra), -1)
        value = F.pad(v, (0, extra))
        y = F.scaled_dot_product_attention(q, k, value, scale=d**-.5, dropout_p=0.)[..., :d]
        return self.out(y.transpose(1, 2).reshape(b, length, self.width))


class RCDBlock(nn.Module):
    def __init__(self, width, heads, channels, ffn_dim, dropout):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.attention = RCDAttention(width, heads, channels)
        self.gate, self.up = nn.Linear(width, ffn_dim), nn.Linear(width, ffn_dim)
        self.down = nn.Linear(ffn_dim, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions, channel_ids):
        x = x + self.attention(self.norm1(x), positions, channel_ids)
        z = self.norm2(x)
        return x + self.dropout(self.down(F.gelu(self.gate(z))*self.up(z)))


class RCDEncoder(nn.Module):
    """Joint attention over C*N channel-patch tokens; returns [B,C,N,D].

    Entirely missing tokens are packed out before attention. Original flattened
    channel-major RoPE positions are preserved, exactly as dense key masking.
    Available channels can differ between Kaggle and LiRA in the same batch.
    """
    def __init__(self, channels=7, patch_length=16, d_model=128, n_heads=4,
                 n_layers=3, ffn_dim=512, dropout=.1, max_patches=256, train_stats=None):
        super().__init__()
        dimensions = (channels, patch_length, d_model, n_heads, n_layers, ffn_dim, max_patches)
        if any(type(v) is not int or v < 1 for v in dimensions) or d_model % (2*n_heads):
            raise ValueError('Positive dimensions and an even dimension per head are required')
        if not 0 <= dropout < 1:
            raise ValueError('Invalid dropout')
        self.config = dict(channels=channels, patch_length=patch_length, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, ffn_dim=ffn_dim, dropout=dropout, max_patches=max_patches)
        self.channels, self.patch_length, self.d_model, self.max_patches = channels, patch_length, d_model, max_patches
        mean = torch.zeros(channels) if train_stats is None else torch.tensor(train_stats['mean'], dtype=torch.float32)
        std = torch.ones(channels) if train_stats is None else torch.tensor(train_stats['std'], dtype=torch.float32)
        if mean.shape != (channels,) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Finite TRAIN normalization required')
        self.register_buffer('mean', mean); self.register_buffer('std', std)
        self.value_embedding = nn.Linear(patch_length, d_model)
        self.valid_embedding = nn.Linear(patch_length, d_model, bias=False)
        nn.init.zeros_(self.valid_embedding.weight)
        self.layers = nn.ModuleList([RCDBlock(d_model, n_heads, channels, ffn_dim, dropout) for _ in range(n_layers)])
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def patchify(self, x, valid_mask=None):
        return normalized_patches(x, valid_mask, self.mean, self.std, self.patch_length, self.max_patches)

    def encode_patches(self, patches, observed):
        b, c, n, _ = patches.shape
        valid = observed.any(-1).reshape(b, c*n)
        tokens = self.value_embedding(patches.masked_fill(~observed, 0))
        tokens = tokens + self.valid_embedding(observed.to(tokens.dtype))
        tokens = tokens.reshape(b, c*n, self.d_model)
        output = torch.zeros_like(tokens)
        patterns, groups = torch.unique(valid, dim=0, return_inverse=True)
        positions = torch.arange(c*n, device=patches.device)
        channel_ids = positions // n
        for index, keep in enumerate(patterns):
            if not keep.any():
                continue
            rows = torch.where(groups == index)[0]
            cols = torch.where(keep)[0]
            z = tokens[rows[:, None], cols[None, :]]
            for layer in self.layers:
                z = layer(z, positions[cols], channel_ids[cols])
            output[rows[:, None], cols[None, :]] = z.to(output.dtype)
        return output.reshape(b, c, n, self.d_model), valid.reshape(b, c, n)

    def forward(self, x, valid_mask=None):
        features, patch_valid = self.encode_patches(*self.patchify(x, valid_mask))
        return dict(features=features, patch_valid=patch_valid)


class RCDPretrainer(nn.Module):
    """15% time-patch masking shared across channels; CE + masked MSE.

    Labels are per-timestep [B,T] binary disturbance labels, with -100 unknown.
    Both heads use the same corrupted forward pass. The projection expands each
    contextual patch into P per-timestep embeddings, following released code.
    Only the encoder transfers to our fresh two-head road model; all pretraining
    heads are retained in the checkpoint for direct anomaly scoring if desired.
    """
    def __init__(self, encoder, d_proj=64, mask_ratio=.15, noise_std=.1, dropout=.1):
        super().__init__()
        if d_proj < 2 or not 0 < mask_ratio < 1 or noise_std < 0:
            raise ValueError('Invalid projection or masking parameters')
        self.encoder, self.mask_ratio, self.noise_std = encoder, mask_ratio, noise_std
        self.config = dict(d_proj=d_proj, mask_ratio=mask_ratio, noise_std=noise_std, dropout=dropout)
        self.projection = nn.Linear(encoder.d_model, encoder.patch_length*d_proj)
        self.reconstruction_head = nn.Sequential(nn.Linear(d_proj, 4*d_proj), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4*d_proj, 4*d_proj), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_proj, 1))
        self.anomaly_head = nn.Sequential(nn.Linear(d_proj, d_proj//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_proj//2, 2))

    def forward(self, x, valid_mask=None, labels=None, *, generator=None, patch_mask=None, noise=None):
        patches, observed = self.encoder.patchify(x, valid_mask)
        b, c, n, p = patches.shape
        available = observed.any(1).any(-1)
        if patch_mask is None:
            count = (available.sum(-1)*self.mask_ratio).long()
            rank = torch.rand((b,n), device=x.device, generator=generator).masked_fill(~available, 2).argsort(-1).argsort(-1)
            patch_mask = (rank < count[:, None]) & available
        if patch_mask.shape != (b,n) or patch_mask.dtype != torch.bool:
            raise ValueError('patch_mask must be boolean [B,N]')
        if noise is None:
            noise = torch.randn(patches.shape, device=x.device, generator=generator)*self.noise_std
        if noise.shape != patches.shape or not torch.isfinite(noise[observed]).all():
            raise ValueError('Finite noise with patch shape required')
        reconstruction_mask = observed & patch_mask[:, None, :, None]
        corrupted = torch.where(reconstruction_mask, noise.detach(), patches)
        features, _ = self.encoder.encode_patches(corrupted, observed)
        local = self.projection(features).reshape(b,c,n,p,-1)
        channel_logits = self.anomaly_head(local).float()
        # Average logits over observed channels, excluding unavailable gyros.
        logits = (channel_logits*observed[..., None]).sum(1) / observed.sum(1).clamp_min(1)[..., None]
        logits = logits.reshape(b,n*p,2)
        targets = patches.detach()[reconstruction_mask]
        predictions = self.reconstruction_head(local[reconstruction_mask]).squeeze(-1).float()
        mse = F.mse_loss(predictions, targets) if targets.numel() else local.sum()*0
        ce = logits.sum()*0
        label_valid = observed.any(1).reshape(b,n*p)
        if labels is not None:
            if labels.shape != (b,n*p) or labels.dtype != torch.long:
                raise ValueError('labels must be int64 [B,T]')
            label_valid = label_valid & (labels != -100)
            if not torch.isin(labels[label_valid], torch.tensor([0,1],device=x.device)).all():
                raise ValueError('Known labels must be binary')
            if label_valid.any():
                ce = F.cross_entropy(logits[label_valid], labels[label_valid])
        else:
            label_valid = torch.zeros_like(label_valid)
        return dict(loss=ce+mse, anomaly_loss=ce, reconstruction_loss=mse, logits=logits,
            prediction=predictions, target=targets, loss_mask=reconstruction_mask,
            label_valid=label_valid, patch_mask=patch_mask)


def load_rcd_encoder(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    encoder = RCDEncoder(**saved['encoder_config'])
    encoder.load_state_dict(saved['encoder_state'])
    return encoder.to(device)
