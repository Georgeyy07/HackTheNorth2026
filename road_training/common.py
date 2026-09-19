"""Shared JSON and SHA-256 helpers. No simulator dependencies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "road_training/data"

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def train_records(manifest, source=None):
    return [r for r in manifest["records"] if r["split"] == "train" and (source is None or r["source"] == source)]
