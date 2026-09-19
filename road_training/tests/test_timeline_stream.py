import torch
from road_training.timeline_stream import RoadTimelineStream


class Toy(torch.nn.Module):
    def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()))
    def forward(self,x,mask):
        values=x[:,::16,0]
        return dict(roughness=values.abs(),disturbance_logit=values,patch_valid=mask[:,::16].any(-1))


def test_arbitrary_chunks_same_committed_timeline_and_provisional_tail():
    config=dict(defect_blend='recent',roughness_blend='latest',onset=.6,offset=.5,roughness_alpha=1.)
    x=torch.randn(16*10+7,7);mask=torch.ones_like(x,dtype=torch.bool)
    mask[16:32]=False
    one=RoadTimelineStream(Toy(),config,'drive');a=one.push(x,mask)
    many=RoadTimelineStream(Toy(),config,'drive');b=[]
    for lo,hi in [(0,3),(3,43),(43,81),(81,len(x))]:b+=many.push(x[lo:hi],mask[lo:hi])
    assert a==b and len(a)==8
    assert a[1]['probability'] is None and a[1]['disturbance'] is None
    assert a[0]['available_s']==.48 and a[0]['start_s']==0
    assert one.finish()['incomplete_samples']==7
    assert [r['target_patch'] for r in one.provisional()]==[8,9]
    assert all(r['status']=='final' for r in a)
    many.reset('new_drive');assert many.session_id=='new_drive' and not many.provisional()
