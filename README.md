# Road training and map replay

PyTorch models for **road roughness (IRI)** and **localized disturbances**, plus a browser replay of timestamped predictions on a map. This repository contains the training/inference code and visualizer from the road-sensor research workspace.

The current deployed model is a four-member, instance-normalized, bidirectional PatchTST ensemble. It consumes 100 Hz acceleration, gyroscope, and speed, and produces two outputs per 160 ms patch: IRI and disturbance probability. The rolling inference wrapper combines overlapping predictions and finalizes them after two subsequent patches (320 ms of observation delay, plus processing/input availability).

## Layout

| Path | Purpose |
| --- | --- |
| `road_training/dataset.py` | Memory-mapped PyTorch dataset; real/synthetic/both filters and fixed splits |
| `road_training/instance_model.py` | Current instance-normalized encoder and two prediction heads |
| `road_training/train_multitask.py` | Shared patch targets, Huber/focal losses, evaluation, and baseline trainer |
| `road_training/mounting_augmentation.py` | Consistent accelerometer/gyro rotation and sensor noise |
| `road_training/checkpoints.py` | Ensemble loading, checksum verification, portable checkpoint receipts |
| `road_training/timeline_stream.py` | Rolling ensemble inference and timestamped provisional/final updates |
| `road_training/export_test_drives.py` | Complete Kaggle/LiRA TEST export with measured GPS and raw signals |
| `road_training/experiments/` | Training recipes, ablations, ensembles, distillation, and research reports |
| `road_training/tools/` | Prepared-corpus import and RoadSens alignment tools |
| `road_viewer/` | FastAPI + Leaflet replay, plots, controls, and browser tests |
| `configs/timeline.json` | Previously selected timeline settings, reused without TEST tuning |
| `docs/` | Data contract, training guide, migration notes, and historical research notes |

## Install

Use Python 3.11+ and Node.js 20+. Install a PyTorch build appropriate for your CUDA environment before training. Run commands from this repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,prepare,viewer,test]'
npm ci --prefix road_viewer
```

For only the dataset/models, `pip install -e .` needs NumPy and PyTorch. The full export also uses the preparation/viewer dependencies. Training experiments use CUDA with BF16 support; CPU inference and the unit tests are supported.

## Use a prepared dataset

Datasets, checkpoints, generated replay exports, and experiment results are external artifacts and are excluded from Git. Supply an existing prepared corpus with `manifest.json` and its record folders. See [the data contract](docs/DATA.md) for structure, labels, orientation, and preparation prerequisites.

```python
from torch.utils.data import DataLoader
from road_training import RoadDataset, InstancePatchTST, InstanceRoadModel

train = RoadDataset(
    'artifacts/data_roadsens_aligned', source='real', split='train',
    window_size=1024, stride=1, return_labels=True,
)
# source may also be 'synthetic' or 'both'.
batch = next(iter(DataLoader(train, batch_size=8)))
model = InstanceRoadModel(InstancePatchTST(max_patches=64))
output = model(batch['x'], batch['mask'])
# output['roughness'], output['disturbance_logit']: [8, 64]
```

Use [the training guide](docs/TRAINING.md) to find the current augmented recipe, the older supervised baselines, and DropPatch/ArcTan/RCD pretraining. The old generic trainer uses TRAIN statistics; the current ensemble recipe uses per-window instance normalization. These are explicitly different experiment configurations.

## Load weights and replay predictions

Place the four trusted member checkpoints under `artifacts/models/`, then create a receipt with relative paths and SHA-256 checksums:

```bash
python -m road_training.checkpoints --output artifacts/models/ensemble.json \
  artifacts/models/seed52.pt artifacts/models/seed53.pt \
  artifacts/models/seed54.pt artifacts/models/seed55.pt
```

```python
from road_training.checkpoints import load_teachers
ensemble = load_teachers('artifacts/models/ensemble.json', device='cuda')
```

To create a new replay export (requires the prepared corpus **and original native sensor/GPS sources** referenced in its manifest):

```bash
python -m road_training.export_test_drives \
  --data artifacts/data_roadsens_aligned \
  --ensemble artifacts/models/ensemble.json \
  --selection configs/timeline.json \
  --output artifacts/test_drive_inference
```

An existing replay export can be viewed directly without weights or a GPU:

```bash
python -m road_viewer.server --export artifacts/test_drive_inference --port 8765
```

Open **http://localhost:8765**. The viewer includes GPS, road colors, disturbance events, accelerometer/gyro plots, playback speed, and seeking. It reconstructs the prediction state using update availability timestamps. See [viewer instructions](road_viewer/README.md).

## Verify

```bash
python -m pytest -q
npm test --prefix road_viewer
```

The unit tests use small fixtures and do not need a downloaded corpus. Browser integration tests additionally need a running viewer and the nine-drive replay export; see its README. Migration validation is recorded in [docs/MIGRATION.md](docs/MIGRATION.md).

## Interpretation

The disturbance output combines manholes, depressions, bumps, cracks, and the applicable dataset-specific disturbances. It is not a dedicated pothole classifier. LiRA supplies measured section IRI; Kaggle does not. LiRA gyroscopes are missing and masked. RoadSens supplies TRAIN-only examples in the existing protocol. Replay colors are model estimates, not verified road-condition ground truth; these evaluations do not establish smartphone deployment reliability.
