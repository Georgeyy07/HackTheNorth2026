"""Causal 100 Hz windows, three-context consensus, saved calibration and Kalman."""
from datetime import datetime, timezone
import math
import numpy as np
from .calibration import calibrated_update
from .client import ENSEMBLE_SHA, validate_prediction
from .filters import AlertPostprocessor, DEFAULT_ALERT_FILTER


def utc(milliseconds):
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class RoadStream:
    def __init__(self, calibration=None):
        self.calibration = calibration
        self.window = np.zeros((1024, 4), dtype=np.float32)
        self.mask = np.zeros((1024, 4), dtype=bool)
        self.pending = []
        self.votes, self.locations = {}, {}
        self.end_patch = -1
        self.last_index = -1
        self.origin = self.utc_origin = self.segment = None
        self.alerts = AlertPostprocessor(**DEFAULT_ALERT_FILTER)

    def prepare(self, samples):
        windows, masks = [], []
        starting_index = self.last_index
        for s in samples:
            t = s.get('time')
            available = s.get('available_at_ms')
            if not finite(t) or not finite(available) or available < 1e12:
                raise ValueError('Samples need time in seconds and available_at_ms in Unix milliseconds')
            values = [s.get(k) for k in ('accel_x', 'accel_y', 'accel_z', 'speed')]
            if any(v is not None and not finite(v) for v in values):
                raise ValueError('Sensor values must be finite numbers or null')
            if values[3] is not None and values[3] < 0:
                raise ValueError('Speed must be nonnegative m/s or null')
            if self.origin is None:
                self.origin, self.utc_origin, self.segment = t, available, s.get('segment', 0)
            if s.get('segment', 0) != self.segment:
                raise ValueError('Sensor segment changed; start a new handshake')
            index = round((t - self.origin) * 100)
            if abs((t - self.origin) * 100 - index) > .05 or index <= self.last_index:
                raise ValueError('Samples must be strictly increasing on a 100 Hz grid')
            if index - self.last_index > 1000:
                raise ValueError('Gap exceeds 10 seconds; start a new handshake')
            if index - starting_index > 4096:
                raise ValueError('Batch spans more than 40.96 seconds')
            for missing in range(self.last_index + 1, index):
                self._append([None] * 4, {}, missing, windows, masks)
            self._append(values, s, index, windows, masks)
            self.last_index = index
        return windows, masks

    def _append(self, values, sample, index, windows, masks):
        self.pending.append(values)
        self.locations.setdefault(index // 16, []).append(sample)
        if len(self.pending) != 16:
            return
        patch = np.asarray(self.pending, dtype=np.float32)
        observed = np.isfinite(patch)
        self.window = np.concatenate((self.window[16:], np.where(observed, patch, 0)))
        self.mask = np.concatenate((self.mask[16:], observed))
        windows.append(self.window.copy()); masks.append(self.mask.copy())
        self.pending = []

    def row(self, target, votes, final):
        valid = bool(votes)
        q = np.average([v[1] for v in votes], axis=0, weights=np.arange(1, len(votes)+1)) if valid else None
        p = float(np.average([v[0] for v in votes], weights=np.arange(1, len(votes)+1))) if valid else None
        grade = int(q[1:].sum() >= .5) + int(q[2] > .5) if valid else None
        row = dict(target_patch=target, target_sample_start=target*16, target_sample_end=(target+1)*16,
                   emitted_after_samples=(self.end_patch+1)*16, start_s=target*.16, end_s=(target+1)*.16,
                   available_s=(self.end_patch+1)*.16, valid=valid, votes=len(votes), is_final=final,
                   status='final' if final else 'provisional', probability=p, disturbance=None,
                   quality_probability=q.tolist() if valid else None, quality_grade=grade,
                   quality_name=['good', 'medium', 'bad'][grade] if valid else None,
                   iri_m_per_km=None, event_id=None, event_transition=None,
                   context_spread=float(np.ptp([v[0] for v in votes])) if valid else None)
        row = self.alerts.update(row)
        row['original_quality_probability'] = row['quality_probability']
        if self.calibration:
            row = calibrated_update(row, self.calibration)
        # A fix must already have arrived in this target patch, not in a later context.
        latitude = longitude = None
        for sample in reversed(self.locations.get(target, [])):
            lat, lon = sample.get('latitude'), sample.get('longitude')
            ts, receipt, available = (sample.get(k) for k in ('gps_timestamp_ms', 'gps_received_at_ms', 'available_at_ms'))
            if (all(finite(v) for v in (lat, lon, ts, receipt, available)) and -90 <= lat <= 90
                    and -180 <= lon <= 180 and ts <= available and 0 <= available-ts <= 3000 and receipt <= available):
                latitude, longitude = lat, lon
                break
        row.update(latitude=latitude, longitude=longitude, observed_at=utc(self.utc_origin+target*160),
                   computed_at=datetime.now(timezone.utc).isoformat(), ensemble_sha256=ENSEMBLE_SHA,
                   calibration_id=self.calibration['id'] if self.calibration else None,
                   calibration_offset=self.calibration['offset'] if self.calibration else 0,
                   settling=any(s.get('settling', False) for s in self.locations.get(target, [])))
        if final:
            self.locations.pop(target, None)
        return row

    async def push(self, samples, client):
        windows, masks = self.prepare(samples)
        updates = []
        for start in range(0, len(windows), 64):
            batch = windows[start:start+64]
            payload = {'windows': np.asarray(batch).tolist(), 'masks': np.asarray(masks[start:start+64]).tolist()}
            result = await client.predict(payload)
            q, p, valid = validate_prediction(result, len(batch))
            for j in range(len(batch)):
                self.end_patch += 1
                for age in range(3):
                    target = self.end_patch-age
                    if target < 0:
                        continue
                    votes = self.votes.setdefault(target, [])
                    if valid[j, 2-age]:
                        votes.append((float(p[j, 2-age]), q[j, 2-age].copy()))
                target = self.end_patch-2
                if target >= 0:
                    updates.append(self.row(target, self.votes.pop(target), True))
                updates.extend(self.row(t, v, False) for t, v in sorted(self.votes.items()))
        return updates
