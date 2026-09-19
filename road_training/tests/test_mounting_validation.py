import pytest
import torch
from torch import nn

from road_training.experiments.mounting_validation import fixed_rotation, FixedMount, SCENARIOS


class Capture(nn.Module):
    def forward(self, x, mask):
        return x, mask


def test_yaw_sign_and_combined_rotation_is_proper():
    rotation = fixed_rotation(yaw=90)
    torch.testing.assert_close(rotation @ torch.tensor([1., 0., 0.]),
                               torch.tensor([0., 1., 0.]), atol=1e-6, rtol=0)
    for scenario in SCENARIOS:
        rotation = fixed_rotation(**{k:scenario[k] for k in ('yaw','pitch','roll')})
        torch.testing.assert_close(rotation.T @ rotation, torch.eye(3), atol=3e-7, rtol=0)
        torch.testing.assert_close(torch.linalg.det(rotation), torch.tensor(1.), atol=3e-7, rtol=0)


def test_fixed_rotation_shared_by_sensors_and_overlapping_windows():
    x=torch.randn(1, 40, 7); x[..., 2]+=9.81; x[..., 3:6]=x[..., :3]*.01
    mask=torch.ones_like(x,dtype=torch.bool)
    model=FixedMount(Capture(),yaw=20,pitch=10)
    whole,valid=model(x,mask)
    window,_=model(x[:,8:24],mask[:,8:24])
    torch.testing.assert_close(whole[:,8:24],window,atol=0,rtol=0)
    torch.testing.assert_close(whole[...,3:6],whole[...,:3]*.01)
    assert torch.equal(whole[...,6],x[...,6]) and torch.equal(valid,mask)


def test_missing_gyro_supported_partial_vector_rejected():
    x=torch.randn(1,16,7); mask=torch.ones_like(x,dtype=torch.bool)
    mask[...,3:6]=False; x[...,3:6]=float('nan')
    transformed,valid=FixedMount(Capture(),roll=10)(x,mask)
    assert torch.isfinite(transformed).all() and not transformed[...,3:6].any()
    assert torch.equal(mask,valid)
    mask[0,1,0]=False
    with pytest.raises(ValueError,match='triads'):
        FixedMount(Capture(),roll=10)(x,mask)


def test_identity_and_bf16_do_not_change_raw_inputs():
    x=torch.randn(2,16,7); mask=torch.ones_like(x,dtype=torch.bool)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        output,_=FixedMount(Capture())(x,mask)
    assert torch.equal(output,x) and output.dtype==torch.float32
