"""Task-aligned teacher targets; unknown ground truth remains unknown."""
import torch
from torch.nn import functional as F


def distillation_loss(student, teacher, targets, datasets, temperature=2.):
    """Bernoulli KL*T² on known detection labels, Huber on known IRI labels.

    teacher logits encode the MEAN of individual softened probabilities.
    Each domain is normalized separately, with Kaggle/RoadSens weights 1/.25.
    No teacher inference or pseudo labels are obtained from VAL or TEST.
    """
    zero = student['roughness'].new_zeros((), dtype=torch.float32)
    dloss, rloss = zero, zero
    valid = student['patch_valid'] & teacher['patch_valid']
    p = teacher['disturbance_logit'].float().detach().sigmoid().clamp(1e-6, 1-1e-6)
    z = student['disturbance_logit'].float()/temperature
    entropy = -p*p.log() - (1-p)*torch.log1p(-p)
    kl = (F.binary_cross_entropy_with_logits(z, p, reduction='none')-entropy)*temperature**2
    for name, weight in {'kaggle': 1., 'roadsens': .25}.items():
        rows = torch.tensor([v == name for v in datasets], device=z.device)[:, None]
        known = valid & targets['disturbance_valid'] & rows
        if known.any():
            dloss = dloss + weight*kl[known].mean()/1.25
    rv = valid & targets['roughness_valid']
    if rv.any():
        rloss = F.smooth_l1_loss(student['roughness'][rv], teacher['roughness'].detach()[rv], beta=1.)
    return dict(total=dloss+.25*rloss, disturbance=dloss, roughness=rloss)
