from types import SimpleNamespace
import numpy as np
import pytest
import torch
from road_training.experiments.streaming_deployment import rolling_teacher_predictions, streaming_predictions
from road_training.streaming_data import ContextWindows
from road_training.dataset import RoadDataset
import json
from road_training.dataset import CHANNELS, KAGGLE_CLASSES, LABELS
from road_training.tools.prepare_data import write_record, training_statistics
from road_training.streaming_model import StreamingRoadModel


def test_context_never_crosses_record_or_split(tmp_path):
    records = []
    for number, split in enumerate(('train', 'val', 'val', 'test')):
        count = 2560 + number * 16
        x = np.full((count, 7), 100. * number, np.float32)
        records.append(write_record(tmp_path,
            dict(id=f'record_{number}', source='real', split=split),
            x, np.ones_like(x, bool), np.arange(count) / 100,
            np.full((count, len(LABELS)), -100, np.int16),
            np.full(count, np.nan, np.float32), {}))
    manifest = dict(sample_rate_hz=100, channels=CHANNELS, records=records,
        label_classes={'kaggle_type': KAGGLE_CLASSES},
        train_statistics={'real': training_statistics(tmp_path, records, 'real')})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    base=RoadDataset(tmp_path,source='real',split='val',return_labels=True,stride=1024)
    assert len(base.records) == 2 and all(r['split'] == 'val' for r in base.records)
    wrapped=ContextWindows(base)
    for record_index in (0,len(base.records)-1):
        first=0 if record_index==0 else base._ends[record_index-1]
        sample=wrapped[first]
        assert not sample['context_mask'][:768].any()
        torch.testing.assert_close(sample['context_x'][768:1792],sample['x'])
        torch.testing.assert_close(sample['context_mask'][768:1792],sample['mask'])
        last=wrapped[base._ends[record_index]-1]
        ends=last['start']-768+np.arange(1,115)*16
        expected=(ends<=base.records[record_index]['samples'])&(ends>0)
        np.testing.assert_array_equal(last['context_available'].numpy(),expected)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA deployment check')
def test_rolling_teacher_target_index_and_future_limit():
    class Toy(torch.nn.Module):
        def forward(self,x,mask):
            return dict(roughness=x[:,::16,0],disturbance_logit=x[:,::16,0],patch_valid=mask[:,::16].any(-1))
    x=np.arange(16*80*7,dtype=np.float32).reshape(16*80,7)
    arrays=dict(x=x,mask=np.ones_like(x,dtype=bool))
    data=SimpleNamespace(records=[{'id':'toy'}],_open=lambda i:arrays)
    output=rolling_teacher_predictions(Toy().cuda(),data)['toy']
    torch.testing.assert_close(output['roughness'][:-2],torch.from_numpy(x[:-32:16,0]))
    assert output['patch_valid'][:-2].all() and not output['patch_valid'][-2:].any()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA deployment check')
def test_continuous_prediction_shift_and_terminal_censoring():
    torch.manual_seed(8)
    model=StreamingRoadModel(width=16,dropout=0,delay_patches=2).cuda().eval()
    rng=np.random.default_rng(9)
    x=rng.normal(size=(80*16,7)).astype(np.float32);mask=np.ones_like(x,dtype=bool)
    data=SimpleNamespace(records=[{'id':'toy'}],_open=lambda i:dict(x=x,mask=mask))
    output=streaming_predictions(model,data)['toy']
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        state=model.initial_state()
        emitted,_=model.stream(torch.from_numpy(x)[None].cuda(),torch.from_numpy(mask)[None].cuda(),state)
    torch.testing.assert_close(output['disturbance_logit'][:-2],emitted['disturbance_logit'][0,2:].float().cpu(),atol=.02,rtol=.03)
    assert not output['patch_valid'][-2:].any()
