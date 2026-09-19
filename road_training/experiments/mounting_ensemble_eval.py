"""Frozen old/new four-member comparison on clean and rotated VAL/TEST."""
import argparse
from contextlib import contextmanager
import gc
from pathlib import Path
import platform

import torch

from road_training.dataset import RoadDataset
from road_training.experiments.mounting_validation import SCENARIOS, FixedMount, compact
from road_training import streaming_evaluation as evaluation
from road_training.experiments.streaming_deployment import timing
from road_training.common import read, write, sha


@contextmanager
def data_context(root):
    previous = evaluation.DATA
    try:
        evaluation.DATA = Path(root)
        yield
    finally:
        evaluation.DATA = previous


def initialize(out, *, baseline, candidate, data_root):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise FileExistsError('Preserve the comparison plan; choose a new directory')
    experiments = Path(__file__).resolve().parent
    package = experiments.parent
    sources = [Path(__file__), experiments / 'mounting_validation.py',
               experiments / 'streaming_deployment.py', package / 'streaming_model.py',
               package / 'streaming_evaluation.py', package / 'streaming_data.py',
               package / 'instance_model.py', package / 'checkpoints.py']
    write(out/'plan.json', dict(data_root=str(Path(data_root).resolve()),
        manifest_sha256=sha(Path(data_root)/'manifest.json'),
        receipts=dict(baseline=str(Path(baseline).resolve()), candidate=str(Path(candidate).resolve())),
        baseline_receipt_sha256=sha(baseline), splits=['val','test'], scenarios=SCENARIOS,
        individual_evaluation='Each new member, clean VAL and TEST; original same-seed scores retained for reference',
        ensemble='All four seeds, equal arithmetic mean of sigmoid probabilities and IRI; fixed threshold 0.5',
        selection='Per-member checkpoints selected using clean VAL only. No new member, weight, threshold or angle selection.',
        test_policy='Freeze both receipts before scoring; evaluate both declared ensembles on the same TEST scenarios regardless of VAL outcome',
        source_sha256={str(p.resolve()):sha(p) for p in sources},
        latency='Synchronized B=1, 1024x7 input; 30 warmups and 200 timings. GPU BF16, CPU FP32 with four threads. Excludes sensor/I-O and observation delay.',
        limitations=['Previously exposed chronological Kaggle split and held-out LiRA road; not a new external phone study.',
                    'RoadSens is TRAIN-only. Rotated copies are sensitivity checks, not independent drives.',
                    'Alignment and augmentation changed together; their individual effects are not separated.',
                    'Offline 10.24-second context results; rolling timeline postprocessing is not evaluated here.',
                    'Yaw +/-30 degrees is an out-of-range stress test.']))


def freeze_receipts(out, plan):
    if sha(plan['receipts']['baseline']) != plan['baseline_receipt_sha256']:
        raise ValueError('Baseline ensemble receipt changed')
    receipts = {}
    for name,path in plan['receipts'].items():
        receipt = read(path)
        if receipt['seeds'] != [52,53,54,55] or len(receipt['checkpoints']) != 4:
            raise ValueError('Expected exactly the four declared seeds')
        for checkpoint,digest in receipt['checkpoints'].items():
            if sha(checkpoint)!=digest:raise ValueError(f'Checkpoint changed: {checkpoint}')
        receipts[name] = dict(path=path, receipt_sha256=sha(path), **receipt)
    path = Path(out)/'frozen_ensembles.json'
    if path.exists():
        if read(path)!=receipts:raise ValueError('Frozen ensemble membership changed')
    else:write(path,receipts)
    return receipts


@torch.inference_mode()
def benchmark(receipts, data_root):
    data = RoadDataset(data_root, source='real', real_dataset='kaggle', split='val',
                       window_size=1024, stride=1024)
    sample = data[0]
    measurements = []
    for device in ('cuda','cpu'):
        torch.set_num_threads(4)
        x,mask=sample['x'][None].to(device),sample['mask'][None].to(device)
        for name,entry in receipts.items():
            model=evaluation.load_teachers(entry['path'],device=device)
            with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
                stats=timing(lambda:model(x,mask),device)
            measurements.append(dict(model=name,device=device,threads=4,**stats))
            del model
            if device=='cuda':torch.cuda.empty_cache()
    return dict(measurements=measurements,cuda=torch.cuda.get_device_name(),
        cpu=platform.processor(),torch_version=torch.__version__,input_shape=[1,1024,7],
        sample_rate_hz=100,patch_length=16,context_seconds=10.24,
        ensemble_parameters=4*841602,method='Model forward only, synchronized, 30 warmups/200 calls; GPU BF16, CPU FP32. No phone or end-to-end delay claim.')


def run(out):
    out=Path(out);plan=read(out/'plan.json')
    for path,digest in plan['source_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'Frozen evaluation source changed: {path}')
    root=Path(plan['data_root'])
    if sha(root/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Dataset changed')
    receipts=freeze_receipts(out,plan)
    torch.set_num_threads(4)
    summary={}
    with data_context(root):
        for split in plan['splits']:
            # Reload and check frozen artifacts at the VAL -> TEST boundary.
            freeze_receipts(out,plan)
            data,loader=evaluation.validation_loader(split)
            summary[split]={}
            for name,entry in receipts.items():
                model=evaluation.load_teachers(entry['path'])
                summary[split][name]={}
                for scenario in plan['scenarios']:
                    write(out/'progress.json',dict(state='evaluating',split=split,model=name,scenario=scenario['name']))
                    wrapped=FixedMount(model,**{k:scenario[k] for k in ('yaw','pitch','roll')}).cuda().eval()
                    result=evaluation.evaluate(wrapped,split=split,loader=loader,data=data)
                    write(out/f'{name}_{split}_{scenario["name"]}.json',result)
                    summary[split][name][scenario['name']]=compact(result)
                if name=='candidate':
                    for seed,member in zip(entry['seeds'],model.models):
                        result=evaluation.evaluate(member,split=split,loader=loader,data=data)
                        write(out/f'candidate_seed{seed}_{split}.json',result)
                del wrapped,model
                gc.collect();torch.cuda.empty_cache()
            write(out/'results.json',summary)
    write(out/'progress.json',dict(state='benchmarking'))
    write(out/'latency.json',benchmark(receipts,root))
    freeze_receipts(out,plan)
    write(out/'progress.json',dict(state='complete',splits=plan['splits'],threshold=.5))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.out)
    except Exception:
        import traceback
        write(args.out/'progress.json',dict(state='failed',error=traceback.format_exc()))
        raise
