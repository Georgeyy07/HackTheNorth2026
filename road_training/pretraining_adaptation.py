"""Linear patch readouts on a truly frozen, deterministic pretrained encoder."""
import torch
from torch import nn
from torch.nn import functional as F


class FrozenLinearRoadModel(nn.Module):
    """Two linear readouts; a softplus link keeps IRI nonnegative.

    Encoder weights, gradients, dropout and normalization stay fixed. Only the
    two Linear layers are trainable. Output/target shapes match the MLP heads.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        width = getattr(encoder, 'feature_channels', encoder.channels) * encoder.d_model
        self.roughness_head = nn.Linear(width, 1)
        self.disturbance_head = nn.Linear(width, 1)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x, valid_mask=None):
        with torch.no_grad():
            encoded = self.encoder(x, valid_mask)
        features = encoded['features']
        b, c, n, d = features.shape
        features = features.permute(0, 2, 1, 3).reshape(b, n, c*d)
        return dict(roughness=F.softplus(self.roughness_head(features).squeeze(-1).float()),
                    disturbance_logit=self.disturbance_head(features).squeeze(-1),
                    patch_valid=encoded['patch_valid'].any(1))


def make_optimizer(model, adaptation, head_lr=1e-4, weight_decay=.01):
    """Explicit groups make the 20:1 learning-rate ratio easy to inspect."""
    if adaptation == 'linear_probe':
        parameters = [p for p in model.parameters() if p.requires_grad]
        groups = [dict(params=parameters, lr=head_lr, name='heads')]
    elif adaptation == 'low_lr':
        heads = list(model.roughness_head.parameters()) + list(model.disturbance_head.parameters())
        groups = [dict(params=list(model.encoder.parameters()), lr=head_lr/20, name='encoder'),
                  dict(params=heads, lr=head_lr, name='heads')]
    elif adaptation == 'full':
        groups = model.parameters()  # Preserve the original control exactly.
    else:
        raise ValueError('Unknown adaptation method')
    return torch.optim.AdamW(groups, lr=head_lr, weight_decay=weight_decay)
