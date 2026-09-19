"""A patch TCN with rolling instance normalization and bounded stream state.

forward() emits a result at the END of each observed 16-sample patch.
With delay_patches=2, result j describes patch j-2 (320 ms after its end).
The delay is label alignment, never a bidirectional operation in the model.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RollingInstanceNorm(nn.Module):
    def __init__(self, patch_length=16, history=16, eps=1e-5):
        super().__init__()
        if patch_length < 1 or history < 1 or eps <= 0:
            raise ValueError('Positive patch length, history and epsilon required')
        self.patch_length, self.history, self.eps = patch_length, history, eps

    def forward(self, x, mask, state=None):
        b, t, c = x.shape
        p = self.patch_length
        if t == 0 or t % p or mask.shape != x.shape or mask.dtype != torch.bool:
            raise ValueError('Whole patches and matching boolean mask required')
        # FP64 moments avoid cancellation from gravity or nearly constant speed.
        values = x.masked_fill(~mask, 0).double().reshape(b, -1, p, c)
        observed = mask.reshape(b, -1, p, c)
        moments = torch.stack((values.sum(2), values.square().sum(2), observed.sum(2)), -1)
        past = moments[:, :0] if state is None else state
        joined = torch.cat((past, moments), 1)
        prefix = F.pad(joined.cumsum(1), (0, 0, 0, 0, 1, 0))
        ends = torch.arange(past.shape[1]+1, joined.shape[1]+1, device=x.device)
        total = prefix[:, ends] - prefix[:, (ends-self.history).clamp_min(0)]
        count = total[..., 2]
        mean = total[..., 0] / count.clamp_min(1)
        variance = (total[..., 1] / count.clamp_min(1) - mean.square()).clamp_min(0)
        scale = (variance+self.eps).sqrt()
        normalized = ((values-mean[:, :, None])/scale[:, :, None]).masked_fill(~observed, 0)
        stats = torch.stack((mean.asinh(), scale.log(), count/(p*self.history)), -1)
        stats = stats.masked_fill((count == 0)[..., None], 0)
        features = torch.cat((normalized.flatten(2), observed.float().flatten(2), stats.flatten(2)), -1).float()
        # A bounded tensor, not a view retaining the entire input allocation.
        new_state = joined[:, -(self.history-1):].clone() if self.history > 1 else joined[:, :0].clone()
        return features, observed.any((2, 3)), new_state


class CausalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.history = 2*dilation
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.linear = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state=None):
        z = self.norm(x).transpose(1, 2)
        past = z.new_zeros(z.shape[0], z.shape[1], self.history) if state is None else state
        joined = torch.cat((past, z), -1)
        out = self.conv(joined).transpose(1, 2)
        result = x + self.dropout(self.linear(F.gelu(out)))
        return result, joined[:, :, -self.history:].clone()


class StreamingRoadModel(nn.Module):
    def __init__(self, channels=7, patch_length=16, width=128,
                 dilations=(1, 2, 4, 8), normalization_history=16,
                 delay_patches=2, dropout=.1):
        super().__init__()
        if delay_patches < 0:
            raise ValueError('Nonnegative output delay required')
        self.config = dict(channels=channels, patch_length=patch_length, width=width,
            dilations=list(dilations), normalization_history=normalization_history,
            delay_patches=delay_patches, dropout=dropout)
        self.channels, self.patch_length, self.delay_patches = channels, patch_length, delay_patches
        self.normalizer = RollingInstanceNorm(patch_length, normalization_history)
        self.embedding = nn.Linear(channels*(2*patch_length+3), width)
        self.blocks = nn.ModuleList(CausalBlock(width, d, dropout) for d in dilations)
        self.norm = nn.LayerNorm(width)
        def head():
            return nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))
        self.roughness_head, self.disturbance_head = head(), head()
        self.receptive_patches = normalization_history + sum(2*d for d in dilations)

    def stream(self, x, mask, state=None):
        """Consume one or more WHOLE patches, returning emitted-time predictions.

        state belongs to exactly this stream. Reset to None at a drive boundary.
        First delay_patches results have no target; after that result j is for
        target j-delay_patches. Do not flush unseen future patches as real data.
        """
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError('Expected [batch,time,configured_channels]')
        history = None if state is None else state['normalizer']
        features, valid, history = self.normalizer(x, mask, history)
        z = self.embedding(features)
        states = []
        for i, block in enumerate(self.blocks):
            z, cache = block(z, None if state is None else state['blocks'][i])
            states.append(cache)
        z = self.norm(z)
        output = dict(roughness=F.softplus(self.roughness_head(z).squeeze(-1).float()),
            disturbance_logit=self.disturbance_head(z).squeeze(-1), patch_valid=valid)
        return output, dict(normalizer=history, blocks=states)

    def forward(self, x, mask):
        return self.stream(x, mask)[0]

    @torch.no_grad()
    def initial_state(self, batch_size=1):
        """Match dataset left-padding at the start of a recording (no real data).

        Call once per drive, in eval mode and the same autocast context as stream.
        This initializes the finite history with masked, unobserved patches.
        """
        parameter = next(self.parameters())
        x = parameter.new_zeros(batch_size, 48*self.patch_length, self.channels)
        return self.stream(x, torch.zeros_like(x, dtype=torch.bool))[1]


def aligned_output(output, context_patches, delay_patches, length=64):
    """Align emitted-time student outputs with the original target patches."""
    start = context_patches+delay_patches
    return {name: value[:, start:start+length] for name, value in output.items()}


class Ensemble(nn.Module):
    """Equal-weight mean of probabilities (not logits) and IRI estimates."""
    def __init__(self, models):
        super().__init__()
        if not models:
            raise ValueError('At least one ensemble member is required')
        self.models = nn.ModuleList(models)

    def forward(self, x, mask, temperature=1.):
        outputs = [model(x, mask) for model in self.models]
        p = torch.stack([(v['disturbance_logit'].float()/temperature).sigmoid() for v in outputs]).mean(0)
        return dict(roughness=torch.stack([v['roughness'].float() for v in outputs]).mean(0),
            disturbance_logit=torch.logit(p.clamp(1e-6, 1-1e-6)),
            patch_valid=torch.stack([v['patch_valid'] for v in outputs]).all(0))
