"""Fixed-delay consensus and one immutable road timeline per drive.

Inputs are the newest rolling WINDOW's patch predictions. Absolute patch IDs
are anchored to the drive start. A patch is finalized once, after delay future
patches. Context disagreement is a diagnostic, not calibrated uncertainty.
"""
import math
import numpy as np


def blend(values, mode):
    """Combine estimates of the SAME patch at progressively later emissions."""
    a=np.asarray(values,dtype=float)
    if mode=='latest':return float(a[-1])
    if mode=='mean':return float(a.mean())
    if mode=='recent':return float(np.average(a,weights=np.arange(1,len(a)+1)))
    raise ValueError('mode must be latest, mean, or recent')


class Timeline:
    def __init__(self, *, delay=2, defect_blend='recent', roughness_blend='recent',
                 onset=.55, offset=.45, roughness_alpha=.5):
        if type(delay) is not int or delay<0 or not 0<=offset<=onset<=1 or not 0<roughness_alpha<=1:
            raise ValueError('Invalid delay, thresholds or roughness smoothing')
        for mode in (defect_blend,roughness_blend):blend([0.],mode)
        self.config=dict(delay=delay,defect_blend=defect_blend,roughness_blend=roughness_blend,
                         onset=onset,offset=offset,roughness_alpha=roughness_alpha)
        self.reset()

    def reset(self):
        self.pending={};self.last_end=-1;self.active=False;self.iri=None
        self.event_start=None;self.event_alert=None;self.event_id=0

    def update(self, end_patch, probability, roughness, valid):
        """Process the full window ending at end_patch, returning finalized rows.

        Calls must advance exactly one globally aligned patch (160 ms). No
        backdated retractions, time-axis interpolation, or growing session list.
        Downstream storage may append each returned row to disk/a database.
        """
        p=np.asarray(probability);r=np.asarray(roughness);v=np.asarray(valid,dtype=bool)
        if end_patch!=self.last_end+1 or p.ndim!=1 or not(p.shape==r.shape==v.shape) or len(p)<=self.config['delay']:
            raise ValueError('Consecutive window end IDs and matching patch arrays required')
        if np.any(~np.isfinite(p[v])) or np.any((p[v]<0)|(p[v]>1)) or np.any(~np.isfinite(r[v])) or np.any(r[v]<0):
            raise ValueError('Valid predictions require finite probability and nonnegative IRI')
        self.last_end=end_patch;delay=self.config['delay']
        for age in range(delay+1):
            target=end_patch-age
            if target<0:continue
            entry=self.pending.setdefault(target,[])
            if v[-1-age]:entry.append((float(p[-1-age]),float(r[-1-age])))
        target=end_patch-delay
        if target<0:return []
        votes=self.pending.pop(target)
        row=self._commit(target,end_patch,votes)
        assert len(self.pending)<=delay
        return [row]

    def _commit(self,target,end,votes,*,truncated=False):
        row=dict(target_patch=target,target_sample_start=target*16,target_sample_end=(target+1)*16,
                 emitted_after_samples=(end+1)*16,status='final' if not truncated else 'terminal_partial',
                 valid=bool(votes),votes=len(votes),event_transition=None,event_id=None)
        if not votes:
            if self.active:
                row.update(event_transition='censored',event_id=self.event_id,
                           event_start_sample=self.event_start,event_end_sample=target*16)
            self.active=False;self.iri=None;self.event_start=None;self.event_alert=None
            row.update(probability=None,disturbance=None,iri_m_per_km=None,quality_grade=None,context_spread=None)
            return row
        p=[v[0] for v in votes];r=[v[1] for v in votes]
        probability=blend(p,self.config['defect_blend'])
        value=blend(r,self.config['roughness_blend'])
        was_active=self.active
        self.active=probability >= (self.config['offset'] if was_active else self.config['onset'])
        if self.active and not was_active:
            self.event_id+=1;self.event_start=target*16;self.event_alert=(end+1)*16
            row['event_transition']='start'
        elif was_active and not self.active:
            row.update(event_transition='end',event_end_sample=target*16)
        if was_active or self.active:
            row.update(event_id=self.event_id,event_start_sample=self.event_start,event_alert_sample=self.event_alert)
        if not self.active:self.event_start=None;self.event_alert=None
        alpha=self.config['roughness_alpha']
        self.iri=value if self.iri is None else alpha*value+(1-alpha)*self.iri
        row.update(probability=probability,disturbance=self.active,iri_raw_m_per_km=value,iri_m_per_km=self.iri,
                   quality_grade=int(np.digitize(self.iri,[2.,4.,6.])),context_spread=float(max(p)-min(p)))
        return row

    def finish(self):
        """Report the uncommitted tail and active event without inventing future.

        No new finalized rows are produced. The latest delay patches remain
        provisional; an active event is right-censored at drive end.
        """
        return dict(uncommitted_patches=sorted(self.pending),active_event_id=self.event_id if self.active else None,
                    event_end_censored=bool(self.active),observed_samples=(self.last_end+1)*16)


def filter_series(probability,roughness,valid,*,onset=.5,offset=.5,roughness_alpha=1.):
    """Same decision state as Timeline, applied to already-fused target rows."""
    active=False;level=None
    labels=np.zeros(len(probability),bool);iri=np.zeros(len(probability),np.float32)
    for i,(p,r,v) in enumerate(zip(probability,roughness,valid)):
        if not v:active=False;level=None;continue
        active=p>=(offset if active else onset)
        level=float(r) if level is None else roughness_alpha*float(r)+(1-roughness_alpha)*level
        labels[i]=active;iri[i]=level
    return labels,iri
