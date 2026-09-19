import pytest

from road_training.experiments import mounting_ensemble_eval as module
from road_training.common import write, sha


def test_evaluation_restores_data_root_on_failure(tmp_path):
    original=module.evaluation.DATA
    with pytest.raises(RuntimeError):
        with module.data_context(tmp_path):
            assert module.evaluation.DATA==tmp_path
            raise RuntimeError('interrupted')
    assert module.evaluation.DATA==original


def test_freeze_receipts_rejects_changed_weights_and_membership(tmp_path):
    paths=[]
    for seed in (52,53,54,55):
        path=tmp_path/f'seed{seed}.pt';path.write_bytes(str(seed).encode());paths.append(path)
    receipt=dict(seeds=[52,53,54,55],checkpoints={str(p):sha(p) for p in paths})
    a,b=tmp_path/'baseline.json',tmp_path/'candidate.json'
    write(a,receipt);write(b,receipt)
    plan=dict(receipts=dict(baseline=str(a),candidate=str(b)),baseline_receipt_sha256=sha(a))
    frozen=module.freeze_receipts(tmp_path,plan)
    assert module.freeze_receipts(tmp_path,plan)==frozen
    paths[0].write_bytes(b'changed')
    with pytest.raises(ValueError,match='Checkpoint changed'):
        module.freeze_receipts(tmp_path,plan)
    paths[0].write_bytes(b'52')
    receipt['seeds']=[52,53,54,56];write(b,receipt)
    with pytest.raises(ValueError,match='four declared seeds'):
        module.freeze_receipts(tmp_path,plan)
