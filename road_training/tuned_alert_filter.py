"""Scalar Kalman filtering with optional innovation-triggered process noise."""
import math
from road_training.alert_filter import ScoreFilter, AlertPostprocessor


def noise_ratio(gain):
    if not 0 < gain < 1:
        raise ValueError("Steady-state gain must be between zero and one")
    return gain * gain / (1 - gain)


class AdaptiveScoreFilter(ScoreFilter):
    """Increase Q on large score innovations, using only the current observation.

    The threshold is in probability-score units even when filtering logits.
    This is an adaptive-noise heuristic, not an optimal or calibrated posterior.
    Small fluctuations use base Q; sharp rises AND falls can use larger Q.
    """
    def __init__(self, *, gain=.9, space="probability", jump=None, fast_gain=.97):
        super().__init__(kind="kalman", space=space, q_over_r=noise_ratio(gain))
        if jump is not None and not 0 < jump <= 1:
            raise ValueError("Innovation threshold must be in (0,1]")
        self.jump = jump
        self.fast_q = noise_ratio(fast_gain) if jump is not None else self.q
        if self.fast_q < self.q:
            raise ValueError("Fast process noise cannot be smaller than base noise")

    def update(self, probability):
        base_q = self.q
        if probability is not None and self.mean is not None and self.jump is not None:
            previous = 1/(1+math.exp(-self.mean)) if self.space == "logit" else self.mean
            if abs(probability-previous) >= self.jump:
                self.q = self.fast_q
        try:
            return super().update(probability)
        finally:
            self.q = base_q


def make_processor(config):
    config = dict(config)
    onset, offset = config.pop("onset"), config.pop("offset")
    if config.pop("original", False):
        return AlertPostprocessor(onset=onset, offset=offset)
    processor = AlertPostprocessor(onset=onset, offset=offset)
    processor.filter = AdaptiveScoreFilter(**config)
    return processor


def process(rows, config):
    processor = make_processor(config)
    return [processor.update(row) for row in rows]
