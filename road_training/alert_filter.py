"""Causal score filtering and hysteresis after same-patch window consensus.

This module never changes IRI, timestamps, GPS, or the original export. A score
is not a calibrated probability after filtering. Kalman covariance describes
the assumed scalar model, not measured neural-network uncertainty.
"""
import math


class ScoreFilter:
    def __init__(self, kind="none", space="probability", alpha=.6, q_over_r=.9):
        if kind not in ("none", "ema", "kalman") or space not in ("probability", "logit"):
            raise ValueError("Unknown filter or score space")
        if not 0 < alpha <= 1 or not math.isfinite(q_over_r) or q_over_r <= 0:
            raise ValueError("Positive noise ratio and alpha in (0,1] required")
        self.kind, self.space, self.alpha, self.q = kind, space, alpha, q_over_r
        self.reset()

    def reset(self):
        self.mean = None
        self.variance = 1.  # R=1; only Q/R determines the scalar gain.

    def update(self, probability):
        if probability is None:
            self.reset()
            return None
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Expected a finite score in [0,1]")
        if self.kind == "none":
            return float(probability)
        p = min(1 - 1e-6, max(1e-6, probability))
        z = math.log(p / (1 - p)) if self.space == "logit" else probability
        if self.mean is None:
            # Initialize from the first observation, without future fitting.
            self.mean, self.variance = z, 1.
        elif self.kind == "ema":
            self.mean += self.alpha * (z - self.mean)
        else:
            prior_variance = self.variance + self.q  # one 160-ms step
            gain = prior_variance / (prior_variance + 1.)
            self.mean += gain * (z - self.mean)
            # Joseph scalar covariance update, with measurement variance R=1.
            self.variance = (1 - gain)**2 * prior_variance + gain**2
        return 1 / (1 + math.exp(-self.mean)) if self.space == "logit" else self.mean


class AlertPostprocessor:
    """Process exported updates in arrival order; only final patches advance state.

    Provisional scores remain raw and explicitly marked. Repeated provisional
    revisions must not be treated as extra temporal observations by the filter.
    Use a fresh object per recording. No minimum duration or future buffer.
    """
    def __init__(self, *, onset=.6, offset=.5, **filter_config):
        if not 0 <= offset <= onset <= 1:
            raise ValueError("Require 0 <= offset <= onset <= 1")
        self.onset, self.offset = onset, offset
        self.filter = ScoreFilter(**filter_config)
        self.last_target = -1
        self.active = False
        self.event_id = 0
        self.event_start = self.event_alert = None

    def update(self, original):
        row = dict(original, original_probability=original.get("probability"))
        if not row["is_final"]:
            row["score_kind"] = "unfiltered_provisional"
            return row
        if row["target_patch"] != self.last_target + 1:
            raise ValueError("Finalized targets must advance exactly once in order")
        self.last_target = row["target_patch"]
        for key in list(row):
            if key.startswith("event_"):
                row.pop(key)
        row.update(event_transition=None, event_id=None, score_kind="filtered_final")
        value = self.filter.update(row["probability"] if row["valid"] else None)
        row["probability"] = value
        if value is None:
            if self.active:
                row.update(event_transition="censored", event_id=self.event_id,
                           event_start_sample=self.event_start, event_end_sample=row["target_sample_start"])
            self.active = False
            self.event_start = self.event_alert = None
            row["disturbance"] = None
            return row
        was_active = self.active
        self.active = value >= (self.offset if was_active else self.onset)
        if self.active and not was_active:
            self.event_id += 1
            self.event_start = row["target_sample_start"]
            self.event_alert = row["emitted_after_samples"]
            row["event_transition"] = "start"
        elif was_active and not self.active:
            row.update(event_transition="end", event_end_sample=row["target_sample_start"])
        if was_active or self.active:
            row.update(event_id=self.event_id, event_start_sample=self.event_start,
                       event_alert_sample=self.event_alert)
        if not self.active:
            self.event_start = self.event_alert = None
        row["disturbance"] = self.active
        return row
