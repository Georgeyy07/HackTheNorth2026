"""Within-dataset MixUp for disturbance labels with missing-label masks.

This is a statistical regularizer, not a physics simulation. LiRA/roughness
examples are never mixed: a convex combination of IMUs has no measured IRI.
"""
import numpy as np
import torch
from torch.nn import functional as F

from road_training.train import focal_from_cross_entropy
from road_training.train_multitask import joint_loss


def mix_disturbance(x, mask, targets, datasets, *, seed, step, alpha=.2, probability=.5):
    """Return mixed inputs and a target-pair recipe; never mix dataset domains.

    Partners share observed channel sets. Missing samples use the intersection
    of their input masks. A mixed patch is supervised only when BOTH original
    patch labels are known. Unknown targets never become negative labels.
    """
    if alpha <= 0 or not np.isfinite(alpha) or not 0 <= probability <= 1:
        raise ValueError("Positive finite alpha and probability in [0,1] required")
    if x.shape != mask.shape or x.ndim != 3 or mask.dtype != torch.bool or len(datasets) != len(x):
        raise ValueError("Expected [B,T,C] values/masks and one dataset per example")
    rng = np.random.default_rng([seed, step, 571])
    partner = np.arange(len(x))
    coefficient = np.ones(len(x), np.float32)
    channel_sets = mask.any(1).cpu().numpy()
    has_roughness = targets["roughness_valid"].any(1).cpu().numpy()
    has_disturbance = targets["disturbance_valid"].any(1).cpu().numpy()
    groups = {}
    for i, name in enumerate(datasets):
        if name in ("kaggle", "roadsens") and not has_roughness[i] and has_disturbance[i]:
            groups.setdefault((name, tuple(channel_sets[i])), []).append(i)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        order = rng.permutation(indices)
        other = np.roll(order, 1)  # No self-pairs.
        use = rng.random(len(order)) < probability
        chosen = order[use]
        partner[chosen] = other[use]
        coefficient[chosen] = rng.beta(alpha, alpha, len(chosen)).astype(np.float32)
    partner = torch.as_tensor(partner, device=x.device)
    coefficient = torch.as_tensor(coefficient, device=x.device)
    active = partner != torch.arange(len(x), device=x.device)
    coefficient = torch.where(active, coefficient.clamp(1e-6, 1-1e-6), coefficient)
    common = mask & mask[partner]
    mixed_mask = torch.where(active[:, None, None], common, mask)
    weight = coefficient[:, None, None]
    mixed_x = (weight*x.masked_fill(~mask, 0) + (1-weight)*x[partner].masked_fill(~mask[partner], 0))
    mixed_x = mixed_x.masked_fill(~mixed_mask, 0)
    return mixed_x, mixed_mask, dict(partner=partner, coefficient=coefficient, active=active)


def mixed_joint_loss(output, targets, recipe, *, gamma=2., alpha=None,
                     roughness_weight=1., disturbance_weight=1.):
    """Blend the two hard-label focal terms; do not threshold a soft label.

    For gamma=0 this is exactly class-weighted soft-label cross entropy.
    For gamma>0 it is the expectation of hard-label focal loss under the mixed
    label distribution. Focal modulation is applied to each class separately.
    Each task is averaged only over its valid patches.
    """
    if recipe is None or not recipe["active"].any():
        return joint_loss(output, targets, gamma=gamma, alpha=alpha,
                          roughness_weight=roughness_weight, disturbance_weight=disturbance_weight)
    partner, weight, active = (recipe[k] for k in ("partner", "coefficient", "active"))
    if targets["roughness_valid"][active].any():
        raise ValueError("MixUp cannot invent a roughness target")
    roughness_only = dict(targets, disturbance_valid=torch.zeros_like(targets["disturbance_valid"]))
    base = joint_loss(output, roughness_only, roughness_weight=roughness_weight)
    valid = targets["disturbance_valid"] & output["patch_valid"]
    valid &= ~active[:, None] | targets["disturbance_valid"][partner]
    dloss = output["disturbance_logit"].new_zeros((), dtype=torch.float32)
    if valid.any():
        logits = output["disturbance_logit"].float()[valid]
        ya = targets["disturbance"][valid]
        yb = targets["disturbance"][partner][valid]
        w = weight[:, None].expand_as(valid)[valid]
        terms = []
        for y in (ya, yb):
            ce = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
            focal = focal_from_cross_entropy(ce, gamma)
            terms.append(focal if alpha is None else focal*alpha[y])
        dloss = (w*terms[0] + (1-w)*terms[1]).mean()
    return dict(total=roughness_weight*base["roughness"] + disturbance_weight*dloss,
                roughness=base["roughness"], disturbance=dloss,
                roughness_valid=base["roughness_valid"], disturbance_valid=valid)


def latent_forward(model, x, mask, recipe):
    """Mix corresponding patch features before the disturbance head only.

    No pooling or temporal shift: [B,C,N,D] features retain patch positions.
    The encoder receives the original sensor-augmented waveforms. Gradients
    reach both partners. Roughness always uses the original features.
    """
    encoded = model.encoder(x, mask)
    features, valid = encoded["features"], encoded["patch_valid"]
    b, c, n, d = features.shape

    def flatten(z):
        return z.permute(0, 2, 1, 3).reshape(b, n, c*d)

    original = flatten(features)
    partner, weight, active = (recipe[k] for k in ("partner", "coefficient", "active"))
    common = valid & valid[partner]
    mixed_valid = torch.where(active[:, None, None], common, valid)
    w = weight[:, None, None, None].to(features.dtype)
    mixed = (w*features + (1-w)*features[partner]).masked_fill(~mixed_valid[..., None], 0)
    return dict(roughness=F.softplus(model.roughness_head(original).squeeze(-1).float()),
                disturbance_logit=model.disturbance_head(flatten(mixed)).squeeze(-1),
                patch_valid=mixed_valid.any(1))
