# Road training and map replay

PyTorch models for **road roughness (IRI)** and **localized disturbances**, plus a browser replay of timestamped predictions on a map. This repository contains the training/inference code and visualizer from the road-sensor research workspace.

The current deployed model is a four-member, instance-normalized, bidirectional PatchTST ensemble. It consumes four 100 Hz inputs: **acceleration X/Y/Z and speed, with no gyroscope**, and produces two outputs per 160 ms patch: IRI and disturbance probability. The rolling inference wrapper combines overlapping predictions and finalizes them after two subsequent patches (320 ms of observation delay, plus processing/input availability). Finalized disturbance scores then pass through the earlier fixed Kalman filter (Q/R = 3.2) and hysteresis (onset 0.70, offset 0.50). Normalization remains per-window instance normalization **with statistics supplied to both heads**.

## Layout

| Path | Purpose |
| --- | --- |
| `road_training/acceleration_speed.py` | Four-input dataset and augmentation; projects canonical stored recordings before training |
| `road_training/dataset.py` | Shared storage reader; real/synthetic/both filters and fixed splits |
| `road_training/instance_model.py` | Current instance-normalized encoder and two prediction heads |
| `road_training/train_multitask.py` | Shared patch targets, Huber/focal losses, evaluation, and baseline trainer |
| `road_training/mounting_augmentation.py` | Mounting rotation and sensor noise, adapted to acceleration only by the four-input loader |
| `road_training/checkpoints.py` | Ensemble loading, checksum verification, portable checkpoint receipts |
| `road_training/timeline_stream.py` | Rolling ensemble inference and timestamped provisional/final updates |
| `road_training/export_test_drives.py` | Complete Kaggle/LiRA TEST export with measured GPS and raw signals |
| `road_training/experiments/` | Training recipes, ablations, ensembles, distillation, and research reports |
| `road_training/tools/` | Prepared-corpus import and RoadSens alignment tools |
| `road_viewer/` | FastAPI + Leaflet replay, plots, controls, and browser tests |
| `configs/timeline.json` | Frozen consensus + fixed Kalman settings, reused without TEST tuning |
| `models/acceleration_speed/` | Four trained, inference-only checkpoints, checksums, and model card |
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

Datasets, training checkpoints, generated replay exports, and experiment results are external artifacts and are excluded from Git. The four small inference-only checkpoints in `models/acceleration_speed/` are included. Supply an existing prepared corpus with `manifest.json` and its record folders. See [the data contract](docs/DATA.md) for structure, labels, orientation, and preparation prerequisites.

```python
from torch.utils.data import DataLoader
from road_training import AccelerationSpeedDataset, InstancePatchTST, InstanceRoadModel

train = AccelerationSpeedDataset(
    'artifacts/data_roadsens_aligned', source='real', split='train',
    window_size=1024, stride=1, return_labels=True,
)
# source may also be 'synthetic' or 'both'.
batch = next(iter(DataLoader(train, batch_size=8)))
model = InstanceRoadModel(InstancePatchTST(channels=4, max_patches=64))
output = model(batch['x'], batch['mask'])
# output['roughness'], output['disturbance_logit']: [8, 64]
```

Use [the training guide](docs/TRAINING.md) to find the current augmented recipe, the older supervised baselines, and DropPatch/ArcTan/RCD pretraining. The old generic trainer uses TRAIN statistics; the current ensemble recipe uses per-window instance normalization. These are explicitly different experiment configurations.

The [ordinal roughness experiment](docs/ORDINAL.md) adds a good/medium/bad model and a PVS importer, with paired regression comparisons and four-seed results. It preserves the current deployment while evaluating the limits of physical IRI labels and weak PVS labels.

## Load weights and replay predictions

The trained four-seed ensemble is included; no retraining or external weight download is needed. Checkpoint hashes and input channel order are checked on load. See [the model card](models/acceleration_speed/README.md).

```python
from road_training.checkpoints import load_teachers
from road_training.timeline_stream import RoadTimelineStream

ensemble = load_teachers('models/acceleration_speed/ensemble.json', device='cuda')
stream = RoadTimelineStream(ensemble, session_id='drive-1')  # Kalman enabled by default.
# new_samples: float32 [N,4], ordered accel_x, accel_y, accel_z, speed.
# new_mask: bool [N,4]; missing speed is zero with a false mask.
final_rows = stream.push(new_samples, new_mask)
provisional_rows = stream.provisional()
# stream.reset('drive-2') at the next drive; do not recreate it per chunk.
```

Provide raw SI values (acceleration including gravity in m/s²; speed in m/s) in the [documented vehicle frame](docs/DATA.md). Do not normalize inputs outside the model. Each final row has filtered `probability`, `original_probability`, IRI, an alert state, and target/availability sample indices. Repeated provisional updates do not advance the Kalman state. `finish()` reports the unfinalized tail without inventing future samples. The filter adds no future buffer; smoothing can delay an alert crossing the threshold.

For newly trained members, `python -m road_training.checkpoints --output artifacts/models/ensemble.json artifacts/models/seed52.pt ...` creates a portable receipt. Seven-input legacy weights are supported explicitly, but they cannot substitute for these trained four-input weights.

To create a new replay export (requires the prepared corpus **and original native sensor/GPS sources** referenced in its manifest):

```bash
python -m road_training.export_test_drives \
  --data artifacts/data_roadsens_aligned \
  --ensemble models/acceleration_speed/ensemble.json \
  --selection configs/timeline.json \
  --output artifacts/test_drive_inference
```

An existing replay export can be viewed directly without weights or a GPU:

```bash
python -m road_viewer.server --export artifacts/test_drive_inference --port 8765
```

Open **http://localhost:8765**. The viewer includes GPS, road colors, disturbance events, raw accelerometer/gyro plots (recorded gyro is retained for inspection only), playback speed, and seeking. It reconstructs the prediction state using update availability timestamps. The new export already contains Kalman-filtered predictions; the viewer does not filter them again or substitute old comparison profiles. See [viewer instructions](road_viewer/README.md).

## Verify

```bash
python -m pytest -q
npm test --prefix road_viewer
```

The unit tests use small fixtures and do not need a downloaded corpus. Browser integration tests additionally need a running viewer and the nine-drive replay export; see its README. Migration validation is recorded in [docs/MIGRATION.md](docs/MIGRATION.md).

## Interpretation

The disturbance output combines manholes, depressions, bumps, cracks, and the applicable dataset-specific disturbances. It is not a dedicated pothole classifier. LiRA supplies measured section IRI; Kaggle does not. LiRA gyroscopes are missing and masked. RoadSens supplies TRAIN-only examples in the existing protocol. Replay colors are model estimates, not verified road-condition ground truth; these evaluations do not establish smartphone deployment reliability.
