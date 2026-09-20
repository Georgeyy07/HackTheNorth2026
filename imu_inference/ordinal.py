"""Three ordered roughness classes, with measured and weak labels kept distinct.

LiRA uses fixed IRI boundaries inspired by FHWA's IRI-only rating (95/170 in/mi).
This is not a complete pavement-condition rating or a calibration of PVS labels.
The two cumulative questions are y > good and y > medium. Ordered logits give
nonnegative probabilities and prevent contradictory threshold predictions.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from imu_inference.instance_model import InstanceRoadModel, InstancePatchTST


NAMES = ('good', 'medium', 'bad')
# Exact unit conversion: one m/km = 63.36 in/mi. Use unrounded measured IRI.
CUTPOINTS = (95. / 63.36, 170. / 63.36)


def iri_classes(iri):
    """Good <95, medium 95..170, bad >170 in/mi; invalid values stay unknown."""
    result = (iri >= CUTPOINTS[0]).long() + (iri > CUTPOINTS[1]).long()
    return result.masked_fill(~torch.isfinite(iri) | (iri < 0), -100)


def ordinal_probabilities(logits):
    cumulative = logits.float().sigmoid()
    return torch.stack((1-cumulative[..., 0], cumulative[..., 0]-cumulative[..., 1],
                        cumulative[..., 1]), -1)


def ordinal_loss(logits, labels, valid, weights=None):
    """Average the two cumulative BCE losses, optionally weighted by section.

    Unknown patches receive no gradient. Weights are observation weights, not
    independent binary class weights that could undo the ordered class meaning.
    """
    if logits.shape != (*labels.shape, 2) or valid.shape != labels.shape:
        raise ValueError('Expected logits [...,2] and matching labels/validity')
    if not valid.any():
        return logits.new_zeros((), dtype=torch.float32)
    y = labels[valid]
    if ((y < 0) | (y > 2)).any():
        raise ValueError('Known ordinal labels must be 0, 1 or 2')
    target = y[..., None] > torch.arange(2, device=y.device)
    loss = F.binary_cross_entropy_with_logits(logits[valid].float(), target.float(), reduction='none').mean(-1)
    if weights is None:
        return loss.mean()
    w = weights[valid].float()
    if not torch.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Observed section weights must be finite and positive')
    return (loss*w).sum()/w.sum()


class OrdinalRoadModel(InstanceRoadModel):
    """Existing encoder/statistics/detector with ordered roughness supervision.

    The original roughness head supplies a positive latent score. In ordinal
    mode its numeric score is NOT claimed to be a calibrated IRI measurement.
    Both modes start with identical trained tensors and identical class outputs.
    PVS has its own learned, ordered cutpoints per training vehicle because its
    acceleration-derived labels are vehicle-relative, not measured IRI classes.
    """
    def __init__(self, encoder, *, mode='ordinal', statistics_mode='both', dropout=.1):
        super().__init__(encoder, statistics_mode=statistics_mode, dropout=dropout)
        if mode not in ('ordinal', 'regression'):
            raise ValueError('mode must be ordinal or regression')
        self.mode = mode
        self.register_buffer('log_cutpoints', torch.tensor(CUTPOINTS).log())
        self.log_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(.25))))
        # Separate annotation calibration for Saveiro, Bravo and Palio.
        self.pvs_start = nn.Parameter(torch.full((3,), math.log(2.)))
        self.pvs_gap = nn.Parameter(torch.full((3,), math.log(math.expm1(math.log(2.)))))
        self.pvs_temperature = nn.Parameter(torch.full((3,), math.log(math.expm1(.25))))

    @classmethod
    def from_regression(cls, saved, *, mode):
        config = saved['config']
        model = cls(InstancePatchTST(**config['encoder_config']), mode=mode, **config['model_config'])
        missing, unexpected = model.load_state_dict(saved['model_state'], strict=False)
        if set(missing) != {'log_cutpoints', 'log_temperature', 'pvs_start', 'pvs_gap', 'pvs_temperature'} or unexpected:
            raise ValueError('Expected an unmodified instance-normalized regression checkpoint')
        return model

    def forward(self, x, valid_mask=None, *, vehicle=None):
        output = super().forward(x, valid_mask)
        score = output.pop('roughness')
        latent = score.clamp_min(1e-6).log()
        logits = (latent[..., None]-self.log_cutpoints)/(F.softplus(self.log_temperature)+1e-6)
        output.update(quality_logits=logits, quality_probability=ordinal_probabilities(logits),
                      quality_class=(score >= CUTPOINTS[0]).long()+(score > CUTPOINTS[1]).long())
        # Training and audits can inspect the latent score without mislabeling it.
        output['quality_score'] = score
        if self.mode == 'regression':
            output['roughness'] = score
        if vehicle is not None:
            if vehicle.shape != (x.shape[0],) or ((vehicle < 0) | (vehicle >= 3)).any():
                raise ValueError('PVS training vehicle indices must be [batch] in 0..2')
            lo = self.pvs_start[vehicle]
            hi = lo+F.softplus(self.pvs_gap[vehicle])+1e-4
            cuts = torch.stack((lo, hi), -1)[:, None]
            scale = F.softplus(self.pvs_temperature[vehicle])[:, None, None]+1e-6
            output['pvs_logits'] = (latent[..., None]-cuts)/scale
        return output
