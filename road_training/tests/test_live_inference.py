import torch
from road_training.live_inference import RoadStream
from road_training.streaming_model import StreamingRoadModel


def test_arbitrary_chunks_delay_missing_target_and_reset():
    torch.manual_seed(55)
    model=StreamingRoadModel(width=16,dropout=0,delay_patches=2).eval()
    x=torch.randn(16*10+7,7);mask=torch.ones_like(x,dtype=torch.bool)
    mask[16:32]=False
    one=RoadStream(model);a=one.push(x,mask)
    many=RoadStream(model);b=[]
    for lo,hi in [(0,3),(3,41),(41,99),(99,len(x))]:b+=many.push(x[lo:hi],mask[lo:hi])
    assert len(a)==len(b)==8
    for aa,bb in zip(a,b):
        assert aa==bb
        assert aa['emitted_after_samples']-aa['target_sample_end']==32
    assert not a[1]['valid'] and a[1]['disturbance_probability'] is None
    assert many.pending.shape==(7,7)
    many.reset();assert many.patches==0 and len(many.pending)==0
    assert many.push(x,mask)==a


def test_rolling_adapter_target_and_bounded_input():
    class Toy(torch.nn.Module):
        channels = 7
        def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()))
        def forward(self,x,mask):
            return dict(roughness=x[:,::16,0],disturbance_logit=x[:,::16,0],patch_valid=mask[:,::16].any(-1))
    stream=RoadStream(Toy())
    x=torch.arange(16*100*7).float().reshape(-1,7)
    output=stream.push(x,torch.ones_like(x,dtype=torch.bool))
    assert len(output)==98 and stream.window.shape==(1,1024,7)
    for i,row in enumerate(output):assert row['iri_m_per_km']==float(x[i*16,0])
