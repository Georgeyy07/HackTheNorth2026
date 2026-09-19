"""Raw 100-Hz samples -> committed patch timeline and event transitions."""
import torch
from road_training.timeline import Timeline, blend
from road_training.streaming_model import StreamingRoadModel
from road_training.checkpoints import model_channels
from road_training.alert_filter import AlertPostprocessor, DEFAULT_ALERT_FILTER


DEFAULT_CONFIG = dict(defect_blend='recent', roughness_blend='latest',
                      onset=.6, offset=.5, roughness_alpha=1.,
                      alert_filter=DEFAULT_ALERT_FILTER)


class RoadTimelineStream:
    """One bounded-state stream for the bidirectional model or ensemble.

    Defaults to three-context consensus followed by the earlier fixed Kalman
    filter on finalized disturbance scores. Explicit legacy configurations
    without alert_filter retain the original unfiltered behavior.
    push returns final rows only; provisional() shows the two unfinished patches.
    Sample indices are relative to session start. Preserve gaps as masked samples
    on the 100-Hz clock, and reset between sessions.
    """
    def __init__(self,model,config=None,session_id=None):
        if isinstance(model,StreamingRoadModel):
            raise ValueError('Overlapping-window consensus requires the bidirectional model')
        self.model=model.eval();self.device=next(model.parameters()).device
        self.channels=model_channels(model)
        config=dict(DEFAULT_CONFIG if config is None else config)
        self.alert_config=config.pop('alert_filter', None)
        self.timeline=Timeline(**config);self.session_id=session_id
        self.reset()

    def reset(self,session_id=None):
        if session_id is not None:self.session_id=session_id
        self.timeline.reset();self.patches=0
        self.alerts=AlertPostprocessor(**self.alert_config) if self.alert_config is not None else None
        self.window=torch.zeros(1,1024,self.channels,device=self.device)
        self.window_mask=torch.zeros_like(self.window,dtype=torch.bool)
        self.pending=self.window[0,:0].clone();self.pending_mask=self.window_mask[0,:0].clone()

    @torch.inference_mode()
    def push(self,x,mask=None):
        x=torch.as_tensor(x,device=self.device,dtype=torch.float32)
        mask=torch.isfinite(x) if mask is None else torch.as_tensor(mask,device=self.device,dtype=torch.bool)
        if x.ndim!=2 or x.shape[1]!=self.channels or mask.shape!=x.shape or not torch.isfinite(x[mask]).all():
            raise ValueError(f'Expected finite observed [samples,{self.channels}] inputs and matching mask')
        x=torch.cat((self.pending,x));mask=torch.cat((self.pending_mask,mask))
        consumed=len(x)//16*16
        self.pending=x[consumed:].clone();self.pending_mask=mask[consumed:].clone()
        rows=[]
        for start in range(0,consumed,16):
            self.window=torch.cat((self.window[:,16:],x[None,start:start+16]),1)
            self.window_mask=torch.cat((self.window_mask[:,16:],mask[None,start:start+16]),1)
            with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                output=self.model(self.window,self.window_mask)
            p=output['disturbance_logit'][0].float().sigmoid().cpu().numpy()
            r=output['roughness'][0].float().cpu().numpy();v=output['patch_valid'][0].cpu().numpy()
            final=self.timeline.update(self.patches,p,r,v);self.patches+=1
            final=[self.postprocess(row) for row in final]
            for row in final:
                row.update(session_id=self.session_id,start_s=row['target_sample_start']/100,
                    end_s=row['target_sample_end']/100,available_s=row['emitted_after_samples']/100)
            rows+=final
        return rows

    def postprocess(self, row):
        """Filter a committed patch once; provisional revisions never advance state."""
        row=dict(row, is_final=row['status']=='final')
        return self.alerts.update(row) if self.alerts is not None else row

    def provisional(self):
        rows=[]
        for target,votes in sorted(self.timeline.pending.items()):
            rows.append(self.postprocess(dict(session_id=self.session_id,target_patch=target,status='provisional',
                votes=len(votes),probability=blend([v[0] for v in votes],self.timeline.config['defect_blend']) if votes else None,
                iri_raw_m_per_km=blend([v[1] for v in votes],self.timeline.config['roughness_blend']) if votes else None)))
        return rows

    def finish(self):
        result=dict(self.timeline.finish(),incomplete_samples=len(self.pending),provisional=self.provisional())
        if self.alerts is not None:
            result.update(active_event_id=self.alerts.event_id if self.alerts.active else None,
                          event_end_censored=self.alerts.active)
        return result
