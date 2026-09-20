"""Frozen experiment recipe; previous pretraining results remain intact."""
from pathlib import Path
from datetime import datetime, timezone
import tarfile
from road_training.common import ROOT, read, write, sha

OUT = ROOT/'reports/rcd_pretraining_20260917'
CORPUS = ROOT/'road_training/data_transfer_v2_mount'
REFERENCE = ROOT/'reports/pretraining_comparison_20260916'
ARMS = {
    'rcd_scratch':dict(method='rcd',pretrained=False),
    'rcd_finetuned':dict(method='rcd',pretrained=True),
    'rcd_linear':dict(method='rcd',pretrained=True,adaptation='linear_probe'),
    'rcd_low_lr':dict(method='rcd',pretrained=True,adaptation='low_lr'),
}


def check_plan():
    plan=read(OUT/'plan.json')
    for path,digest in plan['source_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError(f'Frozen source changed: {path}')
    if sha(CORPUS/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Dataset changed')
    return plan


def write_plan():
    if (OUT/'plan.json').exists():raise FileExistsError('Do not overwrite a frozen plan')
    previous=read(REFERENCE/'plan.json')
    sources=list(Path(__file__).parent.glob('*.py'))
    sources += [ROOT/p for p in ['road_training/rcd.py','road_training/pretrain_rcd.py',
        'road_training/dataset.py','road_training/patchtst.py','road_training/train_multitask.py',
        'road_training/train.py','road_training/pretraining_adaptation.py','road_training/block_sampling.py',
        'road_training/experiments/pretraining_study/report.py','road_training/common.py',
        'road_training/sampling.py','road_training/metrics.py',
        'road_training/domain_evaluation.py']]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),seeds=[42,43,44],arms=ARMS,
        manifest_sha256=sha(CORPUS/'manifest.json'),data_root=str(CORPUS),source_sha256=hashes,
        pretraining=dict(epochs=12,steps_per_epoch=256,batch_size=256,lr=5e-4,weight_decay=1e-5,
            patience=7,source='synthetic',stride=1,window_size=1024,patch_length=16,
            selection='Minimum synthetic VAL CE + masked MSE',normalization='Fixed combined TRAIN statistics',
            masking='15% whole time patches shared across channels; Gaussian replacement sigma .1',
            loss='Unweighted sample-level anomaly cross entropy + masked observed-value MSE; both weights 1',
            sampling='Uniform replacement from corrected physics TRAIN windows; no real labels',
            scheduler='constant',device='cuda',precision='bf16'),
        fine_tuning=previous['fine_tuning'],evaluation=previous['evaluation'],
        reference_comparison_sha256=sha(REFERENCE/'comparison.json'),
        upstream=dict(repository='https://github.com/thu-sail-lab/Time-RCD',
            commit='372bb980426b2f67007311c6f3165ab789c79bef',paper='https://arxiv.org/html/2509.21190v5',
            paper_pdf_sha256=sha(OUT/'time_rcd_v5.pdf')),
        limitations=['This is a compact road adaptation, not a reproduction of the 2.5B-point foundation model.',
            'Physics road labels replace the authors context-dependent generic anomaly corpus.',
            'Fixed combined TRAIN normalization and simulator calibration use real TRAIN information; not strict zero-shot.',
            'Pretraining adds labeled synthetic exposure and compute; primary comparison is against the identical random encoder.',
            'All models keep the prior combined real-VAL loss checkpoint selection and threshold .5.',
            'Historical TEST exposure exists. No TEST-dependent tuning in this study.',
            'Binary disturbance F1 is not defect-type or pothole-specific F1; severe IRI grades remain poorly represented.'])
    plan['evaluation']['split_policy']='Freeze all 12 selected downstream checkpoints after VAL verification, then one fixed TEST sweep'
    write(OUT/'plan.json',plan)
    with tarfile.open(OUT/'source_snapshot.tar.gz','w:gz') as archive:
        for p in hashes:archive.add(ROOT/p,arcname=p)
    print('Saved RCD plan and source snapshot')


if __name__=='__main__':write_plan()
