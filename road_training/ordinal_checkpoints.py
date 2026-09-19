"""Load portable ordinal + disturbance ensembles without experiment runners."""
import torch
from torch import nn

from road_training.acceleration_speed import CHANNELS
from road_training.checkpoints import checkpoint_files
from road_training.common import read
from road_training.instance_model import InstancePatchTST
from road_training.ordinal import CUTPOINTS, NAMES, OrdinalRoadModel


class OrdinalEnsemble(nn.Module):
    """Equal probability averaging; outputs deliberately do not claim numeric IRI."""
    def __init__(self, models):
        super().__init__()
        if not models or any(m.mode != 'ordinal' for m in models):
            raise ValueError('Expected ordinal ensemble members')
        self.models = nn.ModuleList(models)

    def forward(self, x, mask=None):
        outputs = [model(x, mask) for model in self.models]
        quality = torch.stack([o['quality_probability'].float() for o in outputs]).mean(0)
        disturbance = torch.stack([o['disturbance_logit'].float().sigmoid() for o in outputs]).mean(0)
        # Ordinal median, consistent with the reported evaluation protocol.
        classes = (quality[..., 1:].sum(-1) >= .5).long()+(quality[..., 2] > .5).long()
        return dict(quality_probability=quality, quality_class=classes,
                    disturbance_probability=disturbance,
                    disturbance_logit=torch.logit(disturbance.clamp(1e-6, 1-1e-6)),
                    patch_valid=torch.stack([o['patch_valid'] for o in outputs]).all(0))


def load_ordinal_ensemble(receipt, device='cuda'):
    """Verify hashes and input/label contracts before constructing the ensemble."""
    metadata = read(receipt)
    if metadata.get('mode') != 'ordinal' or metadata.get('input_channels') != list(CHANNELS):
        raise ValueError('Expected an ordinal acceleration-XYZ + speed receipt')
    if metadata.get('class_names') != list(NAMES) or metadata.get('cutpoints_m_per_km') != list(CUTPOINTS):
        raise ValueError('Ordinal label definitions do not match the implementation')
    paths = checkpoint_files(receipt)
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('Duplicate ensemble members')
    models = []
    for path in paths:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        config = saved['config']
        if config['model_config'].get('mode') != 'ordinal' or config.get('input_channels') != list(CHANNELS):
            raise ValueError('Checkpoint mode or channel order does not match its receipt')
        model = OrdinalRoadModel(InstancePatchTST(**config['encoder_config']), **config['model_config'])
        geometry = (model.encoder.channels, model.encoder.patch_length, model.encoder.max_patches)
        if geometry != (4, metadata['patch_samples'], metadata['window_samples']//metadata['patch_samples']):
            raise ValueError('Checkpoint patch geometry does not match its receipt')
        model.load_state_dict(saved['model_state'])
        expected = torch.tensor(CUTPOINTS, dtype=model.log_cutpoints.dtype).log()
        if not torch.equal(model.log_cutpoints.cpu(), expected):
            raise ValueError('Checkpoint ordinal cutpoints do not match the receipt')
        models.append(model.to(device).eval())
    return OrdinalEnsemble(models).to(device).eval()
