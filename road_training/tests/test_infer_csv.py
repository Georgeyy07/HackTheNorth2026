import gzip
import json

import numpy as np
import pandas as pd
import pytest
import torch

from road_training.infer_csv import read_recording, export_drive
from road_training.ordinal_stream import OrdinalRoadStream


class FakeOrdinal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, x, mask):
        shape = (len(x), 64)
        p = torch.full(shape, .8, device=x.device)
        q = torch.tensor([.2,.6,.2], device=x.device).expand(*shape, 3)
        valid = mask.reshape(len(x), 64, 16, 4).any(-1).any(-1)
        return dict(quality_probability=q, disturbance_probability=p, patch_valid=valid)


def recording(path, n=167):
    t = np.arange(n)/100
    frame = pd.DataFrame(dict(time_s=t, accel_x=np.sin(t), accel_y=np.zeros(n), accel_z=np.full(n,9.81)))
    frame['latitude_deg'] = np.nan; frame['longitude_deg'] = np.nan
    frame.loc[10, ['latitude_deg','longitude_deg']] = [39.,22.]
    frame.to_csv(path, index=False)
    return frame


def test_causal_resampling_units_missing_speed_and_imu_gaps(tmp_path):
    path = tmp_path/'input.csv'
    pd.DataFrame(dict(timestamp=[0,20,200], ax=[1.,2.,3.], ay=[0.,0.,0.], az=[1.,1.,1.],
                      speed=[np.nan,36.,np.nan])).to_csv(path,index=False)
    samples, gps, x, mask, origin = read_recording(path, time_unit='ms', acceleration_unit='g', speed_unit='km/h')
    assert origin is None and gps.empty
    np.testing.assert_allclose(x[:2,0], 9.80665)
    assert x[2,0] == pytest.approx(2*9.80665)
    assert not mask[:2,3].any() and x[2,3] == 10.
    assert not mask[8:20,:3].any() and (x[8:20,:3] == 0).all()
    assert samples.loc[8:19,'accel_x'].isna().all()


def test_stream_chunking_finalization_kalman_and_gap_reset():
    x = torch.zeros(167,4); x[:,2] = 9.81
    mask = torch.ones_like(x,dtype=torch.bool)
    mask[64:80,:3] = False  # speed still present: no road inference allowed.
    a, b = OrdinalRoadStream(FakeOrdinal()), OrdinalRoadStream(FakeOrdinal())
    all_rows = a.push(x,mask)
    chunked = []
    for start,end in [(0,5),(5,51),(51,66),(66,121),(121,167)]:
        chunked.extend(b.push(x[start:end],mask[start:end]))
    assert all_rows == chunked
    final = [r for r in all_rows if r['is_final']]
    assert [r['target_patch'] for r in final] == list(range(8))
    assert final[0]['available_s'] == pytest.approx(.48)
    assert final[0]['quality_grade'] == 1 and final[0]['iri_m_per_km'] is None
    assert final[0]['event_transition'] == 'start'
    assert not final[4]['valid'] and final[4]['event_transition'] == 'censored'
    assert final[5]['event_transition'] == 'start'
    assert len(a.pending_x) == 7 and set(a.votes) == {8,9}
    assert all(r['score_kind']=='unfiltered_provisional' for r in all_rows if not r['is_final'])
    a.reset()
    assert a.end_patch == -1 and a.alerts.event_id == 0 and not a.votes


def test_export_preserves_every_sample_and_no_future_gps(tmp_path):
    path = tmp_path/'drive.csv'; recording(path)
    out=tmp_path/'export'
    summary=export_drive(path,out,model=FakeOrdinal())
    assert summary['samples']==167 and summary['incomplete_tail_samples']==7
    folder=out/'uploaded_drive'
    assert len(pd.read_parquet(folder/'samples.parquet'))==167
    with gzip.open(folder/'updates.jsonl.gz','rt') as handle:
        rows=[json.loads(line) for line in handle]
    assert all(not r['target_gps_valid'] for r in rows if r['target_patch']==0)
    assert all(r['target_gps_valid'] for r in rows if r['target_patch']==1)
    assert all(r['available_s'] >= r['end_s'] for r in rows)
    assert [r['available_s'] for r in rows] == sorted(r['available_s'] for r in rows)
    assert json.loads((out/'manifest.json').read_text())['score_filter_applied']


def test_predictions_do_not_depend_on_unobserved_future():
    class ContextModel(FakeOrdinal):
        def forward(self, x, mask):
            result = super().forward(x, mask)
            # Depend on every observed sample, so a future leak changes scores.
            p = x[mask].mean().sigmoid()
            result['disturbance_probability'] = p.expand(len(x), 64)
            return result

    x = torch.zeros(1104, 4); x[:, 2] = 9.81
    changed = x.clone(); changed[512:] = 100.
    first = OrdinalRoadStream(ContextModel()).push(x)
    second = OrdinalRoadStream(ContextModel()).push(changed)
    prefix = lambda rows: [r for r in rows if r['emitted_after_samples'] <= 512]
    assert prefix(first) == prefix(second)
    assert first[-1]['probability'] != second[-1]['probability']


def test_mapping_iso_and_future_input_does_not_change_prefix(tmp_path):
    path=tmp_path/'a.csv'; f=recording(path,200)
    f['stamp'] = pd.date_range('2026-01-01T00:00:00Z',periods=200,freq='10ms').astype(str)
    f.to_csv(path,index=False)
    first=read_recording(path,columns={'time_s':'stamp'},time_unit='iso')
    f.loc[100:,'accel_x']=1000.; f.to_csv(path,index=False)
    second=read_recording(path,columns={'time_s':'stamp'},time_unit='iso')
    np.testing.assert_array_equal(first[2][:100],second[2][:100])
    assert first[-1] == 1767225600000000000


@pytest.mark.parametrize('change', ['duplicate','descending','missing_column','invalid_gps'])
def test_invalid_csv_rejected(tmp_path,change):
    path=tmp_path/'drive.csv'; f=recording(path)
    if change=='duplicate': f.loc[1,'time_s']=0.
    if change=='descending': f.loc[1,'time_s']=-1.
    if change=='missing_column': f=f.drop(columns='accel_x')
    if change=='invalid_gps': f.loc[10,'latitude_deg']=100.
    f.to_csv(path,index=False)
    with pytest.raises(ValueError): read_recording(path)
