import torch
from road_training.streaming_model import StreamingRoadModel, RollingInstanceNorm, Ensemble, aligned_output
from road_training.distillation_loss import distillation_loss


def test_stream_matches_batch_and_state_is_bounded():
    torch.manual_seed(7)
    model = StreamingRoadModel(width=24, dropout=0).eval()
    x = torch.randn(2, 16*80, 7); x[...,2] += 9.81
    mask = torch.rand(x.shape) > .15
    mask[...,6] = False
    with torch.no_grad():
        whole = model(x, mask)
        state = None; outputs = []
        for a,b in [(0,16), (16,112), (112,656), (656,1280)]:
            out,state = model.stream(x[:,a:b], mask[:,a:b], state)
            outputs.append(out)
        for key in whole:
            torch.testing.assert_close(torch.cat([o[key] for o in outputs],1), whole[key], atol=2e-6, rtol=1e-5)
        assert state['normalizer'].shape == (2,15,7,3)
        assert [z.shape[-1] for z in state['blocks']] == [2,4,8,16]
        sizes = [z.untyped_storage().nbytes() for z in state['blocks']]
        for _ in range(5):
            _,state = model.stream(x[:,:16], mask[:,:16], state)
        assert sizes == [z.untyped_storage().nbytes() for z in state['blocks']]


def test_future_perturbation_and_gradient_causality():
    torch.manual_seed(8)
    model = StreamingRoadModel(width=16, dropout=0).eval()
    x = torch.randn(1,16*16,7,requires_grad=True); mask = torch.ones_like(x,dtype=torch.bool)
    first = model(x,mask)
    changed = x.detach().clone(); changed[:,8*16:] = 100*torch.randn_like(changed[:,8*16:])
    second = model(changed,mask)
    for key in first:
        torch.testing.assert_close(first[key][:,:8], second[key][:,:8], atol=0,rtol=0)
    first['disturbance_logit'][:,7].sum().backward()
    assert x.grad[:,8*16:].count_nonzero() == 0


def test_masked_values_no_batch_dependence_and_delay_alignment():
    model = StreamingRoadModel(width=16,dropout=0).eval()
    x = torch.randn(2,128,7); mask = torch.rand(x.shape) > .3
    x2 = x.clone(); x2[~mask] = float('nan'); x2[1] *= 999
    with torch.no_grad():
        a,b = model(x,mask),model(x2,mask)
    for key in a:
        torch.testing.assert_close(a[key][0],b[key][0],atol=0,rtol=0)
    out = aligned_output({'x':torch.arange(10)[None]},2,2,length=3)
    assert out['x'].tolist() == [[4,5,6]]


def test_rolling_stats_match_manual_past_only():
    x=torch.arange(1.,33.).reshape(1,16,2)
    mask=torch.ones_like(x,dtype=torch.bool)
    norm=RollingInstanceNorm(patch_length=4,history=2)
    features,_,_=norm(x,mask)
    for j in range(4):
        context=x[:,max(0,(j-1)*4):(j+1)*4]
        expected=(x[:,j*4:(j+1)*4]-context.mean(1,keepdim=True))/(context.var(1,unbiased=False,keepdim=True)+1e-5).sqrt()
        torch.testing.assert_close(features[:,j,:8],expected.flatten(1),atol=1e-6,rtol=1e-6)


def test_ensemble_averages_probabilities():
    class Fake(torch.nn.Module):
        def __init__(self,p):super().__init__();self.p=p
        def forward(self,x,mask):
            return dict(disturbance_logit=torch.logit(torch.tensor([[self.p]])),
                        roughness=torch.tensor([[self.p*4]]),patch_valid=torch.ones(1,1,dtype=torch.bool))
    out=Ensemble([Fake(.1),Fake(.7)])(None,None)
    torch.testing.assert_close(out['disturbance_logit'].sigmoid(),torch.tensor([[.4]]))
    torch.testing.assert_close(out['roughness'],torch.tensor([[1.6]]))


def test_distillation_mask_temperature_teacher_stop_gradient():
    logits=torch.tensor([[.3,-.7]],requires_grad=True)
    teacher_logits=torch.tensor([[.3/2,2.]],requires_grad=True)
    student=dict(disturbance_logit=logits,roughness=torch.ones(1,2,requires_grad=True),patch_valid=torch.ones(1,2,dtype=torch.bool))
    teacher=dict(disturbance_logit=teacher_logits,roughness=torch.zeros(1,2,requires_grad=True),patch_valid=student['patch_valid'])
    target=dict(disturbance_valid=torch.tensor([[True,False]]),roughness_valid=torch.zeros(1,2,dtype=torch.bool))
    loss=distillation_loss(student,teacher,target,['kaggle'])
    assert abs(float(loss['total'].detach())) < 1e-6
    loss['total'].backward()
    assert logits.grad[0,1] == 0
    assert teacher_logits.grad is None and teacher['roughness'].grad is None
    assert student['roughness'].grad is None


def test_finite_context_windows_match_continuous_recording():
    model=StreamingRoadModel(width=16,dropout=0).eval()
    x=torch.randn(1,16*100,7);mask=torch.ones_like(x,dtype=torch.bool)
    with torch.no_grad():
        state=model.initial_state()
        whole,_=model.stream(x,mask,state)
        for start in (0,10,60):
            left=max(0,start-48)
            xx=x.new_zeros(1,52*16,7);mm=torch.zeros_like(xx,dtype=torch.bool)
            count=(start-left)*16
            xx[:,48*16-count:]=x[:,left*16:(start+4)*16]
            mm[:,48*16-count:]=mask[:,left*16:(start+4)*16]
            out=model(xx,mm)
            for key in whole:
                torch.testing.assert_close(out[key][:,48:],whole[key][:,start:start+4],atol=2e-6,rtol=1e-5)
