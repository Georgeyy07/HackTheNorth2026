"""Objective, padding, contextual attention and transfer correctness."""
import math
import torch
from torch.nn import functional as F
from road_training.rcd import RCDAttention, RCDEncoder, RCDPretrainer, load_rcd_encoder
from road_training.patchtst import PatchTSTRoadModel


def setup_module():
    torch.set_num_threads(1)


def test_attention_matches_explicit_upstream_bias_and_gradients():
    torch.manual_seed(9)
    layer=RCDAttention(16,2,3).double()
    with torch.no_grad():layer.bias.normal_()
    x=torch.randn(2,9,16,dtype=torch.double,requires_grad=True)
    positions=torch.tensor([0,1,3,4,5,6,8,10,11]);channels=positions//4
    actual=layer(x,positions,channels)
    split=lambda z:z.reshape(2,9,2,8).transpose(1,2)
    q=split(layer.rotate(layer.q(x),positions));k=split(layer.rotate(layer.k(x),positions));v=split(layer.v(x))
    same=channels[:,None]==channels[None,:]
    bias=torch.where(same[None],layer.bias[1,:,None,None],layer.bias[0,:,None,None])
    weights=((q@k.transpose(-2,-1))/math.sqrt(8)+bias).softmax(-1)
    expected=layer.out((weights@v).transpose(1,2).reshape(2,9,16))
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)
    parameters=(x,*layer.parameters())
    ag=torch.autograd.grad(actual.square().sum(),parameters,retain_graph=True)
    eg=torch.autograd.grad(expected.square().sum(),parameters)
    for a,b in zip(ag,eg):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-8)


def tiny():
    return RCDEncoder(channels=3,patch_length=4,d_model=16,n_heads=2,n_layers=1,ffn_dim=32,dropout=0.)


def test_missing_values_and_empty_sequences_are_isolated():
    torch.manual_seed(1);encoder=tiny().eval()
    x=torch.randn(3,24,3);valid=torch.ones_like(x,dtype=torch.bool)
    valid[0,:,1]=False;valid[1]=False;valid[2,4:8,:]=False
    a=encoder(x,valid);corrupt=x.clone();corrupt[~valid]=float('nan')
    b=encoder(corrupt,valid)
    torch.testing.assert_close(a['features'],b['features'],rtol=0,atol=0)
    assert torch.isfinite(b['features']).all()
    assert (b['features'][~b['patch_valid']]==0).all()
    # Group packing does not change a sample's evaluation features.
    alone=encoder(x[:1],valid[:1])
    torch.testing.assert_close(a['features'][:1],alone['features'])


def test_cross_variate_and_temporal_context_reaches_other_tokens():
    torch.manual_seed(3);encoder=tiny().eval()
    x=torch.randn(1,24,3,requires_grad=True)
    out=encoder(x)['features'][0,0,0].square().sum()
    gradient=torch.autograd.grad(out,x)[0]
    assert gradient[0,12:,2].abs().sum()>0


def test_masked_targets_do_not_leak_and_losses_match_definitions():
    torch.manual_seed(2);model=RCDPretrainer(tiny(),d_proj=8,dropout=0.).eval()
    x=torch.randn(2,24,3,requires_grad=True);valid=torch.ones_like(x,dtype=torch.bool)
    valid[:, :, 1]=False
    labels=torch.zeros(2,24,dtype=torch.long);labels[:,8:12]=1;labels[:,0]=-100
    pm=torch.zeros(2,6,dtype=torch.bool);pm[:,2]=True
    noise=torch.zeros(2,3,6,4)
    a=model(x,valid,labels,patch_mask=pm,noise=noise)
    changed=x.detach().clone();changed[:,8:12]+=99
    b=model(changed,valid,labels,patch_mask=pm,noise=noise)
    torch.testing.assert_close(a['prediction'],b['prediction'],rtol=0,atol=0)
    torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
    assert not torch.equal(a['target'],b['target'])
    torch.testing.assert_close(a['reconstruction_loss'],F.mse_loss(a['prediction'],a['target']))
    torch.testing.assert_close(a['anomaly_loss'],F.cross_entropy(a['logits'][:,1:].reshape(-1,2),labels[:,1:].reshape(-1)))
    torch.testing.assert_close(a['loss'],a['anomaly_loss']+a['reconstruction_loss'])
    grad=torch.autograd.grad(a['prediction'].sum()+a['logits'].sum(),x)[0]
    assert (grad[:,8:12]==0).all() and (grad[:,:,1]==0).all()
    assert a['loss_mask'].sum()==16


def test_unknown_labels_are_excluded_and_random_mask_is_patchwise():
    torch.manual_seed(3)
    model=RCDPretrainer(tiny(),d_proj=8,dropout=0.).eval()
    x=torch.randn(2,80,3);labels=torch.full((2,80),-100,dtype=torch.long)
    result=model(x,labels=labels,generator=torch.Generator().manual_seed(4))
    assert result['patch_mask'].sum(-1).tolist()==[3,3]
    assert result['anomaly_loss']==0 and not result['label_valid'].any()
    assert result['loss_mask'].reshape(2,3,20,4).all(-1).equal(result['patch_mask'][:,None,:].expand(-1,3,-1))
    result['loss'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_checkpoint_transfer_preserves_clean_features_and_supports_both_heads(tmp_path):
    torch.manual_seed(8);encoder=tiny().eval()
    path=tmp_path/'encoder.pt'
    torch.save(dict(encoder_config=encoder.config,encoder_state=encoder.state_dict()),path)
    loaded=load_rcd_encoder(path).eval();x=torch.randn(2,24,3)
    torch.testing.assert_close(encoder(x)['features'],loaded(x)['features'],atol=0,rtol=0)
    road=PatchTSTRoadModel(loaded);out=road(x)
    assert out['roughness'].shape==out['disturbance_logit'].shape==(2,6)
    (out['roughness'].sum()+out['disturbance_logit'].sum()).backward()
    assert road.roughness_head[-1].weight.grad.abs().sum()>0
    assert road.disturbance_head[-1].weight.grad.abs().sum()>0
