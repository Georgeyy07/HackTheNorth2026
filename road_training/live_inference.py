"""Small live adapters: raw samples in, timestamped patch predictions out.

One object per drive. Call reset() at drive boundaries. These adapters retain
only a partial patch and bounded model/history state; they never retain outputs.
"""
from collections import deque
import torch
from road_training.streaming_model import StreamingRoadModel
from road_training.checkpoints import model_channels


class RoadStream:
    """Use a StreamingRoadModel or a bidirectional Ensemble with fixed lookahead.

    Returned target_sample_start/end identify the classified interval. The
    emitted_after_samples field records when the required input became available.
    Device execution time is additional. Input width follows the checkpoint:
    four for acceleration/speed, seven for legacy models. This lower-level
    adapter emits raw scores; use RoadTimelineStream for consensus + Kalman.
    """
    def __init__(self, model, *, delay_patches=2):
        self.model=model.eval()
        self.device=next(model.parameters()).device
        self.channels=model_channels(model)
        self.causal=isinstance(model,StreamingRoadModel)
        self.delay=model.delay_patches if self.causal else delay_patches
        if not 0<=self.delay<64:
            raise ValueError('Delay must be in [0,63] patches')
        self.reset()

    def autocast(self):
        return torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda')

    @torch.inference_mode()
    def reset(self):
        self.pending=torch.zeros(0,self.channels,device=self.device)
        self.pending_mask=torch.zeros(0,self.channels,device=self.device,dtype=torch.bool)
        self.patches=0
        self.validity=deque(maxlen=self.delay+1)
        if self.causal:
            with self.autocast():self.state=self.model.initial_state()
        else:
            self.window=torch.zeros(1,1024,self.channels,device=self.device)
            self.window_mask=torch.zeros_like(self.window,dtype=torch.bool)

    @torch.inference_mode()
    def push(self,x,mask=None):
        x=torch.as_tensor(x,device=self.device,dtype=torch.float32)
        mask=torch.isfinite(x) if mask is None else torch.as_tensor(mask,device=self.device,dtype=torch.bool)
        if x.ndim!=2 or x.shape[1]!=self.channels or mask.shape!=x.shape or not torch.isfinite(x[mask]).all():
            raise ValueError(f'Finite observed values and matching [samples,{self.channels}] masks required')
        x=torch.cat((self.pending,x));mask=torch.cat((self.pending_mask,mask))
        consumed=len(x)//16*16
        self.pending=x[consumed:].clone();self.pending_mask=mask[consumed:].clone()
        results=[]
        for start in range(0,consumed,16):
            patch=x[start:start+16][None];pm=mask[start:start+16][None]
            self.validity.append(bool(pm.any()))
            with self.autocast():
                if self.causal:
                    output,self.state=self.model.stream(patch,pm,self.state);index=0
                else:
                    self.window=torch.cat((self.window[:,16:],patch),1)
                    self.window_mask=torch.cat((self.window_mask[:,16:],pm),1)
                    output=self.model(self.window,self.window_mask);index=63-self.delay
            target=self.patches-self.delay;self.patches+=1
            if target<0:continue
            valid=self.validity[0] and bool(output['patch_valid'][0,index])
            results.append(dict(target_patch=target,target_sample_start=target*16,
                target_sample_end=(target+1)*16,emitted_after_samples=self.patches*16,
                valid=valid,
                disturbance_probability=float(output['disturbance_logit'][0,index].float().sigmoid()) if valid else None,
                iri_m_per_km=float(output['roughness'][0,index]) if valid else None))
        return results
