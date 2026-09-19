import gzip
import json

import numpy as np
import pandas as pd
import torch

from road_training.export_test_drives import available_time, causal_gps, clean_observations, enrich, event_table, infer_drive, interpolation_ready, sample_table, validate_export, write_frame
from road_training.timeline_stream import RoadTimelineStream


CONFIG = dict(defect_blend="recent", roughness_blend="latest", onset=.6,
              offset=.5, roughness_alpha=1.)


class ContextModel(torch.nn.Module):
    channels = 7
    """Depends on all observed context so a lookahead error changes predictions."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, mask):
        v = x.masked_fill(~mask, 0).reshape(-1, 64, 16, 7)
        count = mask.reshape(-1, 64, 16, 7).sum((2, 3))
        average = v.sum((2, 3)) / count.clamp_min(1)
        context = v.sum((1, 2, 3)) / mask.sum((1, 2)).clamp_min(1)
        logits = average + context[:, None]
        return dict(roughness=logits.abs(), disturbance_logit=logits,
                    patch_valid=count > 0)


def test_export_matches_live_stream_and_cannot_see_future():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(16 * 80 + 7, 7)).astype(np.float32)
    mask = np.ones_like(x, bool)
    mask[32:48] = False
    x[~mask] = 0
    updates = []
    rows, finish = infer_drive(ContextModel(), x, mask, CONFIG, updates.append)
    stream = RoadTimelineStream(ContextModel(), CONFIG)
    live = stream.push(x, mask)
    assert [r for r in rows if r["status"] == "final"] == live
    assert len(rows) == 81 and len(live) == 78
    assert rows[-1]["status"] == "provisional_partial"
    assert rows[-1]["target_sample_end"] == len(x)
    assert finish["incomplete_samples"] == 7
    assert [r["target_patch"] for r in updates if r["status"] == "final"] == list(range(78))
    altered = x.copy()
    altered[16 * 70:] += 100
    future, _ = infer_drive(ContextModel(), altered, mask, CONFIG)
    for a, b in zip(rows, future):
        if a["emitted_after_samples"] <= 16 * 70:
            assert a == b


def test_gps_has_no_future_fill_or_stale_coordinates():
    fixes = np.array([[.5, 55., 12.], [1., 56., 13.], [7., 57., 14.]])
    out = causal_gps([0., .5, .9, 1., 4., 4.01, 6.9, 7.], fixes)
    assert out["gps_valid"].tolist() == [False, True, True, True, True, False, False, True]
    assert out["gps_fix_index"].tolist() == [-1, 0, 0, 1, 1, 1, 1, 2]
    assert out["latitude_deg"][2] == 55. and out["longitude_deg"][2] == 12.
    assert np.isnan(out["latitude_deg"][[0, 5, 6]]).all()
    empty = causal_gps([0., 1.], np.empty((0, 3)))
    assert not empty["gps_valid"].any()
    cleaned = clean_observations([[2., 4.], [1., 3.], [1., 5.], [3., np.nan]])
    np.testing.assert_array_equal(cleaned, [[1., 5.], [2., 4.]])


def test_interpolated_input_is_not_available_until_native_endpoint():
    grid = np.array([0., .01, .02, .03])
    ready = interpolation_ready(grid, np.array([0., .02, .071]))
    np.testing.assert_array_equal(ready, [0., .02, .02, .071])
    assert available_time(4, ready) == .071


def test_partial_small_session_reset_missing_and_complete_sample_roundtrip(tmp_path):
    n = 16 * 7 + 9
    x = np.ones((n, 7), np.float32)
    mask = np.ones_like(x, bool)
    mask[:, 3:6] = False
    mask[16:32] = False
    x[~mask] = 0
    ready = np.arange(n) / 100
    fixes = np.array([[.05, 55.7, 12.4], [.7, 55.8, 12.5]])
    native = dict(input_ready_s=ready, fixes=fixes, origin_ns=1600000000000000000,
                  accel_source_time_s=ready, speed_source_time_s=ready)
    updates = []
    rows, finish = infer_drive(ContextModel(), x, mask, CONFIG, updates.append)
    def add(row):
        return enrich(row, "drive", ready, native["origin_ns"], 20., fixes)
    patches = [add(row) for row in rows]
    source_time = 20. + ready
    frame = sample_table(x, mask, source_time, native, patches)
    assert frame.gyro_x.isna().all()
    assert frame.disturbance.iloc[-41:].isna().all()
    assert frame.iloc[-1].probability is not None
    assert frame.loc[16:31, "probability"].isna().all()
    write_frame(tmp_path / "samples", frame)
    with gzip.open(tmp_path / "updates.jsonl.gz", "wt") as handle:
        for row in updates:
            handle.write(json.dumps(add(row), allow_nan=False) + "\n")
    audit = validate_export(tmp_path, x, mask, source_time, patches, native, len(updates))
    assert audit["samples"] == n and audit["final_samples"] == 80
    events = event_table(patches, finish)
    assert events[0]["censor_reason"] == "missing_observations"
    assert events[-1]["end_s"] is None and events[-1]["end_censored"]
    short, _ = infer_drive(ContextModel(), x[:3], mask[:3], CONFIG)
    assert len(short) == 1 and short[0]["target_patch"] == 0
    assert short[0]["status"] == "provisional_partial"


def test_closing_patch_is_not_given_active_event_id():
    rows = []
    for index, (state, transition) in enumerate([(True, "start"), (False, "end"), (False, None)]):
        rows.append(dict(target_patch=index, target_sample_start=index * 16,
                         target_sample_end=(index + 1) * 16, emitted_after_samples=(index + 3) * 16,
                         status="final", valid=True, votes=3, probability=.8 if state else .1,
                         disturbance=state, event_transition=transition,
                         event_id=1 if transition else None, iri_m_per_km=2.,
                         quality_grade=1, context_spread=0.))
    n = 80
    ready = np.arange(n) / 100
    fixes = np.array([[0., 55., 12.]])
    patches = [enrich(r, "drive", ready, 0, 0, fixes) for r in rows]
    events = event_table(patches, dict(total_samples=n))
    assert len(events) == 1 and events[0]["end_s"] == .16
    assert not events[0]["end_censored"] and events[0]["patches"] == 1
