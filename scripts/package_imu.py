"""Package only the original verified ensemble and minimal inference modules."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def package(checkpoint_dir, output):
    expected_path = ROOT/'imu_inference/ensemble.json'
    manifest = json.loads(expected_path.read_text())
    checkpoint_dir, output = Path(checkpoint_dir), Path(output)
    for name, digest in manifest['checkpoints'].items():
        if hashlib.sha256((checkpoint_dir/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Expected original checkpoint: {name}')
    model_dir = output/'data/models';model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(expected_path, model_dir/'ensemble.json')
    for name in manifest['checkpoints']:
        shutil.copy2(checkpoint_dir/name, model_dir/name)
    package_dir = output/'packages/imu_inference';package_dir.mkdir(parents=True, exist_ok=True)
    for name in ['__init__.py', 'patchtst.py', 'instance_model.py', 'ordinal.py', 'model.py']:
        shutil.copy2(ROOT/'imu_inference'/name, package_dir/name)
    print(f'Packaged {len(manifest["checkpoints"])} original checkpoints in {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir', type=Path, default=ROOT/'models/ordinal_pvs')
    p.add_argument('--output', type=Path, default=ROOT/'baseten/imu_ensemble')
    args = p.parse_args();package(args.checkpoint_dir, args.output)
