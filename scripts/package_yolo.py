"""Package the checkpoint selected in vision_inference/manifest.json with the ported YOLO code."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def package():
    manifest = json.loads((ROOT/'vision_inference/manifest.json').read_text())
    weights = ROOT/'models/yolo26/best.pt'
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest['checkpoint_sha256']:
        raise ValueError('Expected the manifest-bound YOLO weights')
    output = ROOT/'baseten/yolo26'
    (output/'data').mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, output/'data/best.pt')
    target = output/'packages/vision_inference';target.mkdir(parents=True, exist_ok=True)
    for name in ('__init__.py','contract.py','model.py','manifest.json'):
        shutil.copy2(ROOT/'vision_inference'/name, target/name)
    print('Packaged verified YOLO26 weights and inference code')


if __name__ == '__main__':
    package()
