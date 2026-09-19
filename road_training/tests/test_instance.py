import pytest
import torch
from road_training.instance_model import instance_normalize, InstancePatchTST, InstanceRoadModel
from road_training.instance_loss import dataset_joint_loss
from road_training.train_multitask import joint_loss


def test_observed_per_window_channel_statistics_and_missing_values():
    x=torch.tensor([[[1.,99.,3.],[2.,float('nan'),3.],[3.,99.,3.]],
                    [[10.,99.,3.],[20.,99.,3.],[30.,99.,3.]]])
    mask=torch.ones_like(x,dtype=torch.bool); mask[:,:,1]=False
    z, valid, stats=instance_normalize(x,mask)
    torch.testing.assert_close(z.mean(1),torch.zeros(2,3),atol=1e-6,rtol=0)
    torch.testing.assert_close(z[:,:,0].square().mean(1),torch.ones(2),atol=2e-5,rtol=0)
    assert not z[:,:,1:].any() and not stats[:,1].any()
    assert torch.isfinite(z).all() and torch.equal(valid,mask)


def test_instances_do_not_influence_each_other_or_use_saved_global_buffers():
    torch.manual_seed(3)
    model=InstanceRoadModel(InstancePatchTST(d_model=16,n_heads=2,n_layers=1,ffn_dim=32,
        patch_length=4,max_patches=2,dropout=0),statistics_mode='both',dropout=0).eval()
    x=torch.randn(2,8,7); mask=torch.ones_like(x,dtype=torch.bool)
    a=model(x,mask)
    x[1]*=1000
    b=model(x,mask)
    for key in a:
        torch.testing.assert_close(a[key][0],b[key][0],atol=0,rtol=0)
    assert torch.equal(model.encoder.mean,torch.zeros(7))
    assert torch.equal(model.encoder.std,torch.ones(7))
    model.encoder.mean.fill_(1000); model.encoder.std.fill_(1000)
    changed_buffers=model(x,mask)
    torch.testing.assert_close(changed_buffers['disturbance_logit'],b['disturbance_logit'],atol=0,rtol=0)
    with pytest.raises(ValueError,match='global'):
        InstancePatchTST(train_stats={'mean':[0]*7,'std':[1]*7})


def test_missing_payload_and_padding_do_not_enter_statistics():
    x=torch.randn(2,12,7,requires_grad=True); mask=torch.rand(x.shape)>.3
    x2=x.detach().clone(); x2[~mask]=float('nan')
    a=instance_normalize(x,mask); b=instance_normalize(x2,mask)
    torch.testing.assert_close(a[0],b[0]);torch.testing.assert_close(a[2],b[2])
    a[0].square().sum().backward()
    assert not x.grad[~mask].any() and torch.isfinite(x.grad).all()


def test_normalization_affine_invariance_and_amplitude_descriptor():
    x=torch.randn(2,32,7)*2
    a=instance_normalize(x); b=instance_normalize(3*x+7)
    torch.testing.assert_close(a[0],b[0],atol=1e-5,rtol=1e-5)
    assert not torch.allclose(a[2],b[2])


def test_hidden_patch_pretraining_rejected_to_prevent_statistic_leakage():
    model=InstancePatchTST(patch_length=4,max_patches=2)
    with pytest.raises(ValueError,match='visible-only'):
        model(torch.randn(2,8,7),patch_mask=torch.zeros(2,7,2,dtype=torch.bool))


def test_dataset_focal_loss_matches_separate_losses_and_ignores_unknowns():
    out=dict(roughness=torch.ones(3,2,requires_grad=True),
        disturbance_logit=torch.tensor([[.7,-1.],[-.2,1.2],[1.,1.]],requires_grad=True),
        patch_valid=torch.ones(3,2,dtype=torch.bool))
    t=dict(roughness=torch.zeros(3,2),roughness_valid=torch.zeros(3,2,dtype=torch.bool),
        disturbance=torch.tensor([[1,0],[0,1],[-100,-100]]),
        disturbance_valid=torch.tensor([[True,True],[True,True],[False,False]]))
    alphas=dict(kaggle=torch.tensor([.5,3.]),roadsens=torch.tensor([2.,.7]))
    losses=dataset_joint_loss(out,t,['kaggle','roadsens','lira'],alphas)
    expected=[]
    for i,name in enumerate(alphas):
        valid=torch.zeros(3,2,dtype=torch.bool);valid[i]=True
        expected.append(joint_loss(out,dict(t,disturbance_valid=valid),alpha=alphas[name])['disturbance'])
    torch.testing.assert_close(losses['disturbance'],sum(expected)/2)
    losses['total'].backward()
    assert out['roughness'].grad is None
    assert not out['disturbance_logit'].grad[2].any()
