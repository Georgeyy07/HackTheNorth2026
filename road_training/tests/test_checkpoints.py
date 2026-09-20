import json

import pytest
import torch

from road_training.checkpoints import checkpoint_files, load_teachers, make_model
from road_training.common import sha


def test_relative_receipt_loads_identical_predictions_and_rejects_changed_weights(tmp_path, monkeypatch):
    config = dict(encoder_config=dict(d_model=16, n_heads=4, n_layers=1,
                                      ffn_dim=32, max_patches=4, dropout=0),
                  model_config=dict(statistics_mode='both'))
    model = make_model(config).eval()
    weights = tmp_path / 'model.pt'
    torch.save(dict(config=config, model_state=model.state_dict()), weights)
    receipt = tmp_path / 'ensemble.json'
    receipt.write_text(json.dumps(dict(checkpoints={'model.pt': sha(weights)})))
    monkeypatch.chdir(tmp_path.parent)  # Resolve against receipt, never working directory.
    loaded = load_teachers(receipt, device='cpu')
    x = torch.randn(1, 64, 7)
    mask = torch.ones_like(x, dtype=torch.bool)
    with torch.no_grad():
        before, after = model(x, mask), loaded(x, mask)
    for key in before:
        torch.testing.assert_close(before[key], after[key])
    weights.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        checkpoint_files(receipt)


def test_empty_ensemble_is_rejected(tmp_path):
    receipt = tmp_path / 'empty.json'
    receipt.write_text('{"checkpoints": {}}')
    with pytest.raises(ValueError, match='at least one'):
        load_teachers(receipt, device='cpu')
