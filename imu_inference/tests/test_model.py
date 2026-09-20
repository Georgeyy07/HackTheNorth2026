"""Exercise the portable, real original weights on CPU."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
import torch
from imu_inference.model import EnsemblePredictor

ROOT = Path(__file__).resolve().parents[2]


def test_original_weights_masking_and_batching():
    torch.set_num_threads(2)
    model = EnsemblePredictor(ROOT/'models/ordinal_pvs')
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2,1024,4)).astype(np.float32)
    x[:,:,2] += 9.81
    x[:,:,3] = 10
    masks = np.ones(x.shape,dtype=bool)
    masks[0, -16:, :3] = False
    result = model.predict(dict(windows=x.tolist(),masks=masks.tolist()))
    assert result['patch_valid'][0] == [True,True,False]
    assert result['patch_valid'][1] == [True,True,True]
    for i in range(2):
        alone = model.predict(dict(windows=x[i:i+1].tolist(),masks=masks[i:i+1].tolist()))
        for k in ('quality_probability','disturbance_probability'):
            np.testing.assert_allclose(np.asarray(result[k])[i],alone[k][0],atol=1e-6)
    manifest = json.loads((ROOT/'models/ordinal_pvs/ensemble.json').read_text())
    for name, checksum in manifest['checkpoints'].items():
        assert hashlib.sha256((ROOT/'models/ordinal_pvs'/name).read_bytes()).hexdigest() == checksum
    with pytest.raises(ValueError,match='shape'):
        model.predict({'windows':[[0,0,9.81,0]]})
