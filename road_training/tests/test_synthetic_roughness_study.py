import numpy as np
import pandas as pd
import torch

from road_training.instance_model import InstancePatchTST, InstanceRoadModel
from road_training.experiments.synthetic_roughness_study import set_head_training, non_roughness_state, assert_frozen, metrics


def test_roughness_learning_leaves_detector_predictions_and_weights_identical():
    torch.manual_seed(121)
    model = InstanceRoadModel(InstancePatchTST(d_model=8, n_heads=2, n_layers=1, ffn_dim=16, max_patches=4), statistics_mode='both')
    set_head_training(model)
    frozen = non_roughness_state(model)
    x = torch.randn(2, 64, 7); x[:, :, 2] += 9.8
    mask = torch.ones_like(x, dtype=torch.bool)
    before = model(x, mask)
    detector = before['disturbance_logit'].detach().clone()
    original_iri = before['roughness'].detach().clone()
    optimizer = torch.optim.AdamW(model.roughness_head.parameters(), lr=.01)
    optimizer.zero_grad(set_to_none=True)
    (before['roughness'] - 4.).square().mean().backward()
    optimizer.step()
    assert_frozen(model, frozen)
    after = model(x, mask)
    assert torch.equal(detector, after['disturbance_logit'])
    assert not torch.equal(original_iri, after['roughness'])
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith('roughness_head.'))


def test_section_metrics_do_not_weight_long_sections_or_repeated_passes_more():
    cache = dict(meta=pd.DataFrame(dict(recording=['slow'] * 90 + ['fast'] * 10,
                                       section=[0] * 90 + [1] * 10,
                                       spatial=['road'] * 100, domain=['lira'] * 100)),
                 y=torch.tensor([1.] * 90 + [4.] * 10))
    result = metrics(np.array([3.] * 90 + [4.] * 10), cache)
    assert result['section_mae'] == 1.
    assert np.isclose(result['patch_mae'], 1.8)
    assert result['sections'] == 2
