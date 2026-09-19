"""Four fresh instances of the frozen successful recipe; no TEST access.

Reuse its training functions without changing any historical implementation.
Run: python -m road_training.experiments.ensemble_teachers --init / --run
"""
import argparse
import os
from pathlib import Path
import traceback
from road_training.experiments import instance_study as study
from road_training.experiments import instance_roughness as adaptation
from road_training.common import read, write, sha

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/ensemble_causal_20260917'
TEACHERS = OUT / 'teachers'
SEEDS = [52, 53, 54, 55]
ARM = 'instance_detection_focus'


def initialize():
    study.check_plan()
    OUT.mkdir(parents=True, exist_ok=True)
    TEACHERS.mkdir(exist_ok=True)
    if (TEACHERS/'plan.json').exists():
        raise FileExistsError('Frozen teacher plan already exists')
    plan = read(study.OUT/'plan.json')
    plan.update(arms={ARM: plan['arms'][ARM]}, seeds=SEEDS,
                purpose='Four fresh independent seeds, equal-weight probability/IRI ensemble',
                final_selection='All four members; no member selection. No TEST until causal comparison frozen.')
    plan['source_sha256'][str(Path(__file__).resolve())] = sha(__file__)
    write(TEACHERS/'plan.json', plan)
    head_plan = read(adaptation.OUT/'plan.json')
    head_plan.update(seeds=SEEDS, trigger='Always refine each roughness head; epoch zero eligible')
    write(TEACHERS/'head_plan.json', head_plan)
    write(OUT/'protocol.json', dict(teacher_seeds=SEEDS,
        teachers='Four fresh runs of existing successful real-only augmented recipe',
        ensemble='Equal arithmetic mean of probabilities and IRI; fixed threshold 0.5',
        selection='Validation only; TEST only after all finalists are frozen',
        causal='Patch-based causal dilated temporal convolutions with finite streaming buffers',
        planned_delay_patches=[0, 2], sample_rate_hz=100, patch_samples=16,
        normalization='Teacher full-instance; causal student past/current-instance statistics only',
        experiments='Matched supervised and teacher-distilled students; validate delay and confirm seeds',
        deployment='Benchmark available CUDA BF16 and CPU FP32; distinguish compute and observation delay',
        limitations=plan['limitations']))


def run():
    study.OUT = TEACHERS
    adaptation.BASE = TEACHERS
    adaptation.OUT = TEACHERS/'roughness_head_adaptation'
    adaptation.OUT.mkdir(exist_ok=True)
    plan = read(TEACHERS/'head_plan.json')
    for seed in SEEDS:
        write(OUT/'teacher_progress.json', dict(state='joint_training', seed=seed, pid=os.getpid()))
        study.train(ARM, seed)
        write(OUT/'teacher_progress.json', dict(state='roughness_head', seed=seed, pid=os.getpid()))
        adaptation.train(seed, ARM, plan)
    checkpoints = [adaptation.OUT/f'{ARM}_seed{s}/best.pt' for s in SEEDS]
    write(OUT/'teacher_checkpoints.json', dict(seeds=SEEDS,
        checkpoints={str(p): sha(p) for p in checkpoints}, test_read=False))
    write(OUT/'teacher_progress.json', dict(state='complete', pid=os.getpid()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.init:
        initialize()
    elif args.run:
        import fcntl
        with (OUT/'teacher.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run()
            except Exception:
                write(OUT/'teacher_failure.json', dict(error=traceback.format_exc()))
                raise
    else:
        parser.error('Choose --init or --run')
