"""Deterministic mounting-error validation, separate from random TRAIN augmentation."""
import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from road_training.dataset import RoadDataset
from road_training.experiments.instance_study import make_model
from road_training.mounting_augmentation import axis_angle
from road_training.streaming_data import ContextWindows
from road_training import streaming_evaluation as evaluation
from road_training.train_multitask import supervised_windows
from road_training.common import read, write, sha

SCENARIOS = [dict(name='clean', yaw=0., pitch=0., roll=0., in_range=True)]
SCENARIOS += [dict(name=f'yaw_{angle:+d}', yaw=float(angle), pitch=0., roll=0.,
                   in_range=abs(angle)<=20) for angle in (-30, -20, -10, 10, 20, 30)]
SCENARIOS += [dict(name=f'{axis}_{angle:+d}', yaw=0.,
                  pitch=float(angle) if axis=='pitch' else 0.,
                  roll=float(angle) if axis=='roll' else 0., in_range=True)
              for axis in ('pitch', 'roll') for angle in (-10, 10)]
SCENARIOS += [dict(name=f'yaw_{angle:+d}_pitch_{angle//2:+d}', yaw=float(angle),
                   pitch=float(angle/2), roll=0., in_range=True) for angle in (-20, 20)]


def fixed_rotation(yaw=0., pitch=0., roll=0.):
    """Rz(yaw) @ Ry(pitch) @ Rx(roll), around nominal model axes, in degrees."""
    if any(not math.isfinite(v) for v in (yaw, pitch, roll)):
        raise ValueError('Finite mounting angles required')
    matrices = [axis_angle(torch.tensor([axis]), torch.tensor([math.radians(angle)]))[0]
                for axis, angle in (([0., 0., 1.], yaw), ([0., 1., 0.], pitch),
                                    ([1., 0., 0.], roll))]
    return matrices[0] @ matrices[1] @ matrices[2]


class FixedMount(nn.Module):
    """One constant frame error for every window/record; no random/noise transform."""
    def __init__(self, model, *, yaw=0., pitch=0., roll=0.):
        super().__init__()
        self.model = model
        self.register_buffer('rotation', fixed_rotation(yaw, pitch, roll))

    def forward(self, x, mask):
        # Do not change supervised coverage to accommodate a missing component.
        for offset in (0, 3):
            valid = mask[..., offset:offset+3]
            if (valid.any(-1) & ~valid.all(-1)).any():
                raise ValueError('Fixed-angle validation requires complete or entirely missing XYZ triads')
        with torch.autocast(device_type=x.device.type, enabled=False):
            rotated = x.masked_fill(~mask, 0).float().clone()
            for offset in (0, 3):
                rotated[..., offset:offset+3] = rotated[..., offset:offset+3] @ self.rotation.float().T
        return self.model(rotated, mask)


def initialize(out, checkpoints, data_root):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise FileExistsError('Preserve validation protocol; use another directory')
    sources = [Path(__file__), Path(__file__).resolve().parents[2] / 'road_training/streaming_evaluation.py',
               Path(__file__).resolve().parents[2] / 'road_training/mounting_augmentation.py']
    write(out/'plan.json', dict(split='val', scenarios=SCENARIOS,
        checkpoints={name:str(Path(path).resolve()) for name,path in checkpoints.items()},
        data_root=str(Path(data_root).resolve()), manifest_sha256=sha(Path(data_root)/'manifest.json'),
        source_sha256={str(p.resolve()):sha(p) for p in sources},
        threshold=.5, comparison='Same seed, architecture, raw units, task labels, evaluation coverage and fixed rotations',
        selection='Joint checkpoint selected on clean VAL F1; IRI head selected on clean VAL patch MAE. No selection on rotated cases.',
        finality='Validation diagnostics only; no TEST reads or automatic model promotion',
        rotation='Constant per scenario across all windows and recordings; same rotation for accel/gyro before normalization',
        limitations=['Only one seed; changing alignment and augmentation together does not isolate their individual effects.',
                    'Synthetic coordinate rotations are not independent drives or proof of real-device robustness.',
                    'RoadSens remains TRAIN-only; Kaggle and LiRA retain their original validation splits.',
                    'Yaw +/-30 is an extrapolation stress test beyond the training yaw range.']))


def compact(result):
    d = result['disturbance']['classification']['per_class']['disturbance']
    return dict(f1=d['f1'], precision=d['precision'], recall=d['recall'],
        iri_patch_mae=result['roughness']['mae'], iri_section_mae=result['sections']['mae'],
        ordinal_f1=result['sections']['ordinal']['macro_f1_present_classes'])


def run(out):
    out = Path(out); plan = read(out/'plan.json')
    for path,digest in plan['source_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'Validation implementation changed: {path}')
    root = Path(plan['data_root'])
    if sha(root/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Validation dataset changed')
    torch.set_num_threads(4)
    # Freeze every checkpoint before computing any angle-specific metrics.
    receipt = {name:dict(path=path,sha256=sha(path)) for name,path in plan['checkpoints'].items()}
    if (out/'checkpoints.json').exists():
        if receipt!=read(out/'checkpoints.json'):raise ValueError('A frozen checkpoint changed')
    else:write(out/'checkpoints.json', receipt)
    data = RoadDataset(root, split='val', source='real', stride=1024, return_labels=True)
    selected, _, _ = supervised_windows(data, 16)
    loader = DataLoader(Subset(ContextWindows(data), selected.indices), batch_size=256,
                        shuffle=False, num_workers=2, pin_memory=True)
    previous = evaluation.DATA
    summary = {}
    try:
        evaluation.DATA = root
        for name,entry in receipt.items():
            saved = torch.load(entry['path'], map_location='cpu', weights_only=False)
            model = make_model(saved['config']).cuda().eval()
            model.load_state_dict(saved['model_state'])
            summary[name] = {}
            for scenario in plan['scenarios']:
                wrapped = FixedMount(model, **{k:scenario[k] for k in ('yaw','pitch','roll')}).cuda().eval()
                result = evaluation.evaluate(wrapped, loader=loader, data=data)
                write(out/f'{name}_{scenario["name"]}.json', result)
                summary[name][scenario['name']] = compact(result)
                write(out/'progress.json', dict(state='evaluating', model=name, scenario=scenario['name']))
            del model, wrapped
            torch.cuda.empty_cache()
    finally:
        evaluation.DATA = previous
    write(out/'results.json', summary)
    write(out/'progress.json', dict(state='complete', split='val', checkpoints=receipt))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.out)
