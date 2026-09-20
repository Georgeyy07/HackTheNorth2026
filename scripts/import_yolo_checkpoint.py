"""Download one exact Baseten training artifact and bind its hash/class names.

Requires BASETEN_API_KEY and the YOLO runtime dependencies. A missing checkpoint
is an error; this command never silently substitutes another training run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--filename', default='best.pt')
    args = p.parse_args()
    key = os.environ.get('BASETEN_API_KEY')
    if not key:
        p.error('Set BASETEN_API_KEY')
    base = f'https://api.baseten.co/v1/training_projects/{args.project}/jobs/{args.job}'
    with httpx.Client(headers={'Authorization': f'Api-Key {key}'}, timeout=30) as api:
        candidates, page_token = [], None
        while True:
            params = {'page_size': 100}
            if page_token:
                params['page_token'] = page_token
            response = api.get(base+'/checkpoint_files', params=params)
            response.raise_for_status()
            page = response.json()
            candidates.extend(page.get('presigned_urls', []))
            page_token = page.get('next_page_token')
            if not page_token:
                break
    # Truss returns entries with the checkpoint path and a presigned URL.
    matches = [item for item in candidates if any(
        isinstance(value,str) and (value == args.filename or value.endswith('/'+args.filename))
        for name,value in item.items() if 'url' not in name.lower())]
    if len(matches) != 1:
        raise SystemExit(f'Expected exactly one {args.filename} in job {args.job}; found {len(matches)} (total files: {len(candidates)})')
    item = matches[0]
    url = next((v for k,v in item.items() if 'url' in k.lower() and isinstance(v,str) and v.startswith('https://')),None)
    if not url:
        raise SystemExit('Checkpoint entry has no HTTPS download URL')
    # The signed artifact URL authorizes the download; do not forward the API key.
    try:
        response = httpx.get(url, timeout=120, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError:
        raise SystemExit('Checkpoint download failed') from None
    data = response.content
    os.environ.setdefault('YOLO_AUTOINSTALL','false')
    from ultralytics import YOLO
    with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
        tmp.write(data);tmp.flush()
        model = YOLO(tmp.name)
        names = {str(k):str(v).lower() for k,v in model.names.items()}
    if not set(names.values()) <= {'pothole','crack'} or 'pothole' not in names.values():
        raise SystemExit('Expected a pothole/crack model')
    target = ROOT/'models/yolo26/best.pt'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(data)
    manifest_path = ROOT/'vision_inference/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(training_project_id=args.project, training_job_id=args.job,
                    source_path=args.filename, checkpoint_sha256=hashlib.sha256(data).hexdigest(), classes=names,
                    note='Checkpoint downloaded from the explicitly selected Baseten training job.')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('training_job_id','checkpoint_sha256','classes')},indent=2))


if __name__ == '__main__':
    main()
