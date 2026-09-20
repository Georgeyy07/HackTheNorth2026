"""Freezing includes dropout and optimizer state; low LR affects encoder only."""
import pytest
import torch

from road_training.arctan import ArcTanEncoder
from road_training.patchtst import PatchTST, PatchTSTRoadModel
from road_training.pretraining_adaptation import FrozenLinearRoadModel, make_optimizer


@pytest.mark.parametrize('kind',[PatchTST,ArcTanEncoder])
def test_frozen_linear_encoder_is_deterministic_and_bitwise_unchanged(kind):
    torch.set_num_threads(1);torch.manual_seed(4)
    encoder=kind(channels=3,patch_length=4,d_model=16,n_heads=2,n_layers=1,ffn_dim=32,dropout=.5)
    model=FrozenLinearRoadModel(encoder).train()
    assert not encoder.training
    assert all(not p.requires_grad for p in encoder.parameters())
    x=torch.randn(2,32,3)
    first=model(x);second=model(x)
    for key in first:torch.testing.assert_close(first[key],second[key],atol=0,rtol=0)
    before={k:v.clone() for k,v in encoder.state_dict().items()}
    optimizer=make_optimizer(model,'linear_probe')
    included={id(p) for g in optimizer.param_groups for p in g['params']}
    assert all(id(p) not in included for p in encoder.parameters())
    head_before=model.disturbance_head.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad();out=model(x)
        (out['roughness'].square().mean()+out['disturbance_logit'].square().mean()).backward()
        optimizer.step()
    assert all(p.grad is None for p in encoder.parameters())
    assert not torch.equal(head_before,model.disturbance_head.weight)
    for key,value in encoder.state_dict().items():assert torch.equal(value,before[key])
    model.eval();model.train();assert not encoder.training


def test_low_lr_groups_are_complete_disjoint_and_twenty_to_one():
    encoder=PatchTST(channels=3,patch_length=4,d_model=16,n_heads=2,n_layers=1,ffn_dim=32)
    model=PatchTSTRoadModel(encoder)
    optimizer=make_optimizer(model,'low_lr')
    assert [g['lr'] for g in optimizer.param_groups]==[5e-6,1e-4]
    groups=[{id(p) for p in g['params']} for g in optimizer.param_groups]
    assert not groups[0]&groups[1]
    assert groups[0]=={id(p) for p in encoder.parameters()}
    assert groups[0]|groups[1]=={id(p) for p in model.parameters()}
    # Equal artificial gradients at zero weights produce 20x different steps.
    ep=next(encoder.parameters());hp=next(model.roughness_head.parameters())
    with torch.no_grad():ep.zero_();hp.zero_()
    ep.grad=torch.ones_like(ep);hp.grad=torch.ones_like(hp)
    optimizer.step()
    torch.testing.assert_close(hp.flatten()[0],20*ep.flatten()[0])
