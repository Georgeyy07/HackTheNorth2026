import json

import pytest
import torch

from road_training.common import sha
from road_training.instance_model import InstancePatchTST
from road_training.ordinal import CUTPOINTS, NAMES, OrdinalRoadModel
from road_training.ordinal_checkpoints import load_ordinal_ensemble


def package(tmp_path):
    models, files = [], {}
    for seed in (1, 2):
        torch.manual_seed(seed)
        encoder = InstancePatchTST(channels=4, d_model=8, n_heads=2, n_layers=1,
                                  ffn_dim=16, max_patches=4, dropout=0)
        model = OrdinalRoadModel(encoder, mode='ordinal', dropout=0).eval()
        config = dict(encoder_config=encoder.config, model_config=dict(mode='ordinal', dropout=0),
                      input_channels=['accel_x', 'accel_y', 'accel_z', 'speed'])
        path = tmp_path/f'seed{seed}.pt'
        torch.save(dict(config=config, model_state=model.state_dict()), path)
        files[path.name] = sha(path)
        models.append(model)
    receipt = tmp_path/'ensemble.json'
    receipt.write_text(json.dumps(dict(mode='ordinal', checkpoints=files,
        input_channels=config['input_channels'], class_names=NAMES, cutpoints_m_per_km=CUTPOINTS,
        sample_rate_hz=100, patch_samples=16, window_samples=64)))
    return models, receipt


def test_portable_loading_matches_probability_ensemble_and_masks(tmp_path):
    originals, receipt = package(tmp_path)
    model = load_ordinal_ensemble(receipt, device='cpu')
    x, mask = torch.randn(2, 64, 4), torch.ones(2, 64, 4, dtype=torch.bool)
    mask[0, :16] = False
    mask[1, :, 3] = False
    with torch.inference_mode():
        result = model(x, mask)
        values = [m(x, mask) for m in originals]
    q = torch.stack([o['quality_probability'] for o in values]).mean(0)
    d = torch.stack([o['disturbance_logit'].sigmoid() for o in values]).mean(0)
    torch.testing.assert_close(result['quality_probability'], q)
    torch.testing.assert_close(result['disturbance_probability'], d)
    assert torch.equal(result['quality_class'], (q[..., 1:].sum(-1) >= .5).long()+(q[..., 2] > .5).long())
    assert not result['patch_valid'][0, 0] and result['patch_valid'][1].all()
    assert 'roughness' not in result and 'quality_score' not in result
    assert all(not m.training for m in model.modules())


def test_corrupt_package_rejected_before_loading(tmp_path):
    _, receipt = package(tmp_path)
    with (tmp_path/'seed2.pt').open('ab') as f:
        f.write(b'corruption')
    with pytest.raises(ValueError, match='SHA-256'):
        load_ordinal_ensemble(receipt, device='cpu')


@pytest.mark.parametrize('key,value', [('mode', 'regression'), ('class_names', ['bad','medium','good']),
    ('input_channels', ['speed','accel_x','accel_y','accel_z']), ('cutpoints_m_per_km', [1., 3.])])
def test_wrong_contract_rejected(tmp_path, key, value):
    _, receipt = package(tmp_path)
    metadata = json.loads(receipt.read_text()); metadata[key] = value
    receipt.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_ordinal_ensemble(receipt, device='cpu')
