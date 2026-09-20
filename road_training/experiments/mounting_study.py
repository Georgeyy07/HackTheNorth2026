"""New training recipe with phone-mount augmentation; old experiments frozen.

Uses the successful instance-normalized real-only trainer, changing just its
training augmentation in a scoped adapter. Validation remains unaugmented.
No TEST evaluation or automatic promotion of a checkpoint happens here.
"""
import argparse
from contextlib import contextmanager
from functools import partial
import os
from pathlib import Path
import tarfile
import traceback

from road_training.experiments import instance_study as study
from road_training.mounting_augmentation import perturb
from road_training.common import read, write, sha

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'reports/mounting_augmentation_20260919/training'
DEFAULT_DATA = ROOT / 'road_training/data_roadsens_aligned'
ARM = 'mounting_detection_focus'


def initialize(out, *, data_root=DEFAULT_DATA, seed=72, yaw_degrees=20., tilt_degrees=10., probability=.75,
               epochs=24, steps_per_epoch=256):
    """Freeze the new recipe and its sources without modifying the old plan."""
    # Validate the same options as the augmenter before creating a run folder.
    import torch
    from road_training.mounting_augmentation import mounting_rotations
    x = torch.zeros(1, 1, 7); x[..., 2] = 9.81
    mounting_rotations(x, torch.ones_like(x, dtype=torch.bool), seed=seed, step=0,
        yaw_degrees=yaw_degrees, tilt_degrees=tilt_degrees, probability=probability)
    if epochs < 1 or steps_per_epoch < 1:
        raise ValueError('epochs and steps_per_epoch must be positive')
    plan = study.check_plan()
    parent = study.OUT/'plan.json'
    data_root = Path(data_root).resolve()
    manifest = read(data_root/'manifest.json')
    if 'roadsens_alignment' not in manifest:
        raise ValueError('Prepare the aligned RoadSens dataset first with tools.align_roadsens')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise FileExistsError('Preserve existing mounting plan; choose a new output directory')
    augmentation = dict(yaw_degrees=yaw_degrees, tilt_degrees=tilt_degrees, probability=probability)
    plan.update(arms={ARM: plan['arms']['instance_detection_focus']}, seed=seed,
        epochs=epochs, steps_per_epoch=steps_per_epoch, mounting_augmentation=augmentation,
        data_root=str(data_root), manifest_sha256=sha(data_root/'manifest.json'),
        parent_plan_sha256=sha(parent),
        augmentation='TRAIN only: fixed-per-window gravity-axis yaw and horizontal tilt, shared accel/gyro; existing noise/bias',
        final_selection='Single development run, clean VAL checkpoint selection; no TEST access or checkpoint promotion')
    for key in ('screen_seed', 'confirmation_seeds', 'finalists'):
        plan.pop(key, None)
    # The inherited update-based stopping rule continues to work for short smoke runs.
    plan['early_stopping']['minimum_updates'] = min(1536, epochs * steps_per_epoch)
    new_files = ['road_training/mounting_augmentation.py', 'road_training/experiments/mounting_study.py',
                 'road_training/tools/align_roadsens.py',
                 'road_training/tests/test_mounting_augmentation.py', 'road_training/tests/test_roadsens_alignment.py']
    plan['source_sha256'].update({str(ROOT/p): sha(ROOT/p) for p in new_files})
    plan['limitations'] += [
        'Mount augmentation is not frame calibration; source-specific axis conventions remain.',
        'Mean total acceleration approximates vertical and can be biased by driving dynamics.',
        'This recipe does not simulate a moving/loose mount or promise arbitrary-orientation robustness.',
        'A subsequent roughness-head refinement must also be checked for orientation sensitivity.']
    write(out/'plan.json', plan)
    with tarfile.open(out/'sources.tar.gz', 'w:gz') as archive:
        for path in plan['source_sha256']:
            archive.add(path, arcname=str(Path(path).relative_to(ROOT)))
    return plan


@contextmanager
def configured_study(out):
    """Reuse the frozen trainer with explicit, temporarily replaced settings."""
    out = Path(out).resolve()
    previous_out, previous_data, previous_perturb = study.OUT, study.DATA, study.perturb
    try:
        study.OUT = out
        study.DATA = Path(read(out/'plan.json')['data_root'])
        plan = study.check_plan()
        study.perturb = partial(perturb, **plan['mounting_augmentation'])
        yield plan
    finally:
        study.OUT, study.DATA, study.perturb = previous_out, previous_data, previous_perturb


def run(out):
    import fcntl
    out = Path(out).resolve()
    with (out/'runner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with configured_study(out) as plan:
                write(out/'progress.json', dict(state='training', seed=plan['seed'], pid=os.getpid()))
                study.train(ARM, plan['seed'])
                write(out/'progress.json', dict(state='complete', seed=plan['seed'], test_evaluated=False))
        except Exception:
            write(out/'progress.json', dict(state='failed', error=traceback.format_exc(), pid=os.getpid()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--init', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--yaw-degrees', type=float, default=20.)
    parser.add_argument('--tilt-degrees', type=float, default=10.)
    parser.add_argument('--probability', type=float, default=.75)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--steps-per-epoch', type=int, default=256)
    args = parser.parse_args()
    if args.init:
        initialize(args.out, data_root=args.data_root, seed=args.seed, yaw_degrees=args.yaw_degrees,
            tilt_degrees=args.tilt_degrees, probability=args.probability,
            epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)
    else:
        run(args.out)
