"""Stateless, ordinary PyTorch inference for the original four-model ensemble."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .instance_model import InstancePatchTST
from .ordinal import CUTPOINTS, NAMES, OrdinalRoadModel

CHANNELS = ['accel_x', 'accel_y', 'accel_z', 'speed']


class EnsemblePredictor:
    def __init__(self, directory, device='cpu'):
        directory = Path(directory)
        manifest_path = directory/'ensemble.json'
        manifest = json.loads(manifest_path.read_text())
        if (manifest['mode'] != 'ordinal' or manifest['input_channels'] != CHANNELS
                or manifest['class_names'] != list(NAMES)
                or manifest['cutpoints_m_per_km'] != list(CUTPOINTS)
                or len(manifest['checkpoints']) != 4):
            raise ValueError('Expected the original four-member ordinal ensemble')
        self.ensemble_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.device = torch.device(device)
        self.models = []
        for name, expected in manifest['checkpoints'].items():
            path = directory/name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Ensemble checkpoint checksum mismatch')
            saved = torch.load(path, map_location='cpu', weights_only=True)
            config = saved['config']
            if config['input_channels'] != CHANNELS or config['model_config']['mode'] != 'ordinal':
                raise ValueError('Checkpoint input contract mismatch')
            model = OrdinalRoadModel(InstancePatchTST(**config['encoder_config']), **config['model_config'])
            model.load_state_dict(saved['model_state'], strict=True)
            model.requires_grad_(False).to(self.device).eval()
            self.models.append(model)

    @torch.inference_mode()
    def predict(self, payload):
        x = np.asarray(payload['windows'], dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (1024, 4) or not 1 <= len(x) <= 64:
            raise ValueError('windows must have shape [1..64,1024,4]')
        mask = np.isfinite(x)
        if 'masks' in payload:
            supplied = np.asarray(payload['masks'])
            if supplied.shape != x.shape or supplied.dtype != np.bool_:
                raise ValueError('masks must be boolean and match windows')
            mask &= supplied
        if np.isinf(x).any():
            raise ValueError('Infinite sensor values are invalid')
        x = torch.as_tensor(np.where(mask, x, 0), device=self.device)
        mask = torch.as_tensor(mask, device=self.device)
        # Plain FP32 PyTorch. No compilation, quantization, or optimization pass.
        outputs = [model(x, mask) for model in self.models]
        quality = torch.stack([o['quality_probability'] for o in outputs]).mean(0)
        defect = torch.stack([o['disturbance_logit'].float().sigmoid() for o in outputs]).mean(0)
        valid = torch.stack([o['patch_valid'] for o in outputs]).all(0)
        valid &= mask[..., :3].all(-1).reshape(len(x), 64, 16).all(-1)
        return {'quality_probability': quality[:, -3:].cpu().tolist(),
                'disturbance_probability': defect[:, -3:].cpu().tolist(),
                'patch_valid': valid[:, -3:].cpu().tolist(),
                'patch_indices': [61, 62, 63], 'ensemble_sha256': self.ensemble_sha256}
