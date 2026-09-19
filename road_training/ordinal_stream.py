"""Bounded rolling ordinal inference with delayed consensus and alert filtering."""
import numpy as np
import torch

from road_training.alert_filter import AlertPostprocessor, DEFAULT_ALERT_FILTER


class OrdinalRoadStream:
    """100-Hz vehicle-frame XYZ + speed -> timestamped ordinal patch updates.

Three progressively later contexts vote on each target. Only finalized
targets advance the fixed Kalman filter. The last two patches remain
provisional; reset creates a fresh drive and filter state.
"""
    def __init__(self, model):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.reset()

    def reset(self):
        self.window = torch.zeros(1, 1024, 4, device=self.device)
        self.mask = torch.zeros_like(self.window, dtype=torch.bool)
        self.pending_x = self.window[0, :0].clone()
        self.pending_mask = self.mask[0, :0].clone()
        self.votes = {}
        self.end_patch = -1
        self.alerts = AlertPostprocessor(**DEFAULT_ALERT_FILTER)

    def row(self, target, votes, final):
        valid = bool(votes)
        q = np.average([v[1] for v in votes], axis=0, weights=np.arange(1, len(votes)+1)) if valid else None
        p = float(np.average([v[0] for v in votes], weights=np.arange(1, len(votes)+1))) if valid else None
        grade = int(q[1:].sum() >= .5)+int(q[2] > .5) if valid else None
        row = dict(target_patch=target, target_sample_start=target*16, target_sample_end=(target+1)*16,
                   emitted_after_samples=(self.end_patch+1)*16, start_s=target*.16, end_s=(target+1)*.16,
                   available_s=(self.end_patch+1)*.16, valid=valid, votes=len(votes), is_final=final,
                   status='final' if final else 'provisional', probability=p, disturbance=None,
                   quality_probability=q.tolist() if valid else None, quality_grade=grade,
                   quality_name=['good', 'medium', 'bad'][grade] if valid else None,
                   iri_m_per_km=None, event_id=None, event_transition=None,
                   context_spread=float(np.ptp([v[0] for v in votes])) if valid else None)
        return self.alerts.update(row)

    @torch.inference_mode()
    def push(self, x, mask=None):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        mask = torch.isfinite(x) if mask is None else torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if x.ndim != 2 or x.shape[1] != 4 or mask.shape != x.shape or not torch.isfinite(x[mask]).all():
            raise ValueError('Expected finite observed [samples,4] inputs and matching mask')
        x = torch.cat((self.pending_x, x)); mask = torch.cat((self.pending_mask, mask))
        consumed = len(x)//16*16
        self.pending_x, self.pending_mask = x[consumed:].clone(), mask[consumed:].clone()
        updates = []
        for start in range(0, consumed, 16):
            self.window = torch.cat((self.window[:,16:], x[None,start:start+16]), 1)
            self.mask = torch.cat((self.mask[:,16:], mask[None,start:start+16]), 1)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
                output = self.model(self.window, self.mask)
            p = output['disturbance_probability'][0].float().cpu().numpy()
            q = output['quality_probability'][0].float().cpu().numpy()
            valid = output['patch_valid'][0].cpu().numpy()
            # Speed alone must not turn an IMU outage into a road prediction.
            valid &= self.mask[0, :, :3].all(-1).reshape(64, 16).all(-1).cpu().numpy()
            self.end_patch += 1
            for age in range(3):
                target = self.end_patch-age
                if target < 0:
                    continue
                votes = self.votes.setdefault(target, [])
                if valid[-1-age]:
                    votes.append((float(p[-1-age]), q[-1-age].copy()))
            target = self.end_patch-2
            if target >= 0:
                updates.append(self.row(target, self.votes.pop(target), True))
            updates.extend(self.row(t, v, False) for t, v in sorted(self.votes.items()))
        return updates
