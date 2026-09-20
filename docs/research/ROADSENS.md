> Historical research note from the source workspace. Paths, commands, and run results may refer to earlier experiments. See [the current training guide](../TRAINING.md) and [migration notes](../MIGRATION.md) before running a command.

# RoadSens-4M IMU integration

For new orientation-aware training, use `data_roadsens_aligned` and the
[mounting recipe](MOUNTING.md). It rotates RoadSens into nominal forward/left/up
coordinates and preserves this original import for historical experiments.

Prepared root: `road_training/data_with_roadsens`. This extends
`data_transfer_v2_mount` with RoadSens TRAIN recordings. Existing records are
linked, and the previous manifest is saved alongside the new one. Historical
datasets, normalizers and checkpoints are unchanged.

```python
from road_training.dataset import RoadDataset

root = "road_training/data_with_roadsens"
train = RoadDataset(root, source="real", split="train", stride=1,
                    return_labels=True)  # Kaggle + LiRA + RoadSens
roadsens = RoadDataset(root, source="real", real_dataset="roadsens",
                       split="train", stride=1, return_labels=True)
mixed = RoadDataset(root, source="both", split="train", stride=1,
                    return_labels=True)  # Also includes existing synthetic data
```

`real_dataset` can be `all`, `kaggle`, `lira`, or `roadsens`. `source` remains
`real`, `synthetic`, or `both`. Each filter has matching TRAIN-only channel
statistics. The model applies these fixed statistics for z-score normalization;
the dataset returns raw SI values and availability masks.

## Inputs and targets

Input and mask shapes are `[1024, 7]`. Six channels are observed:
`totalAcceleration_x/y/z` in m/s², including gravity, and calibrated
`gyroscope_x/y/z` in rad/s. The seventh speed channel is zero **with its mask
false**. The release has no vehicle speed in its combined CSVs; weather wind
speed is not vehicle speed. Recorded device axes are preserved. In particular,
the observed gravity is mostly on device Y in these files; the importer does
not assume the paper's nominal vehicle-axis description is the recorded frame.

These CSVs are already on a continuous 100 Hz grid. There is no new
interpolation, backward filling, per-record normalization or inferred speed.
Nonfinite sensor values are masked. We cannot recover the timing of the
publisher's upstream averaging from these combined files.

The binary disturbance labels supervise:

- Bump/pothole intervals whose text run has exactly two button markers, one at
  each endpoint. A button's press duration is **not** an event duration.
- Normal road samples from the five explicitly designated normal recordings.

Blank labels inside anomaly recordings, unclosed trailing events and invalid
IMU samples remain `-100` (unknown). Both `defect` and
`defect_assumed_normal` use this conservative policy for RoadSens. No weak
negative is invented from blank text. Original bump/pothole distinctions are
retained in `roadsens_type.npy` and event metadata, but not mapped onto the
Kaggle taxonomy. The standard two-head trainer consumes their binary union.
Its majority patch targets have `[batch, 64]` shape for 16-sample patches.

RoadSens provides **no measured IRI or ordinal road-quality ground truth**.
All roughness masks stay false; these records supervise only disturbance.
LiRA and synthetic records continue to supervise roughness.

## Counts and qualifications

The combined archive contains 101 sensor CSVs and one misplaced summary table.
Session 99’s complete continuous CSV is recovered from the isolated archive.
Five sensor CSVs repeat other recordings exactly. After removing those copies
and the summary table, the import contains:

| Item | Count |
|---|---:|
| Unique recordings | 97 |
| 100 Hz samples | 647,762 |
| Duration | 107.96 minutes |
| Closed annotated intervals | 317 |
| Positive sample labels | 273,553 |
| Explicit normal sample labels | 44,046 |
| Unknown sample labels | 330,163 |
| Usable 1,024-sample windows, stride 1 | 549,285 |
| Windows containing supervision | 426,640 |

Ninety-one recordings are long enough for 1,024-sample windows; the other six
remain available for shorter-window consumers. Positive labels are 86.1% of
known labels, so this is a selected-event collection, not natural driving
prevalence. Some manual intervals cover extended rough stretches rather than
one impact. Adding it has **not yet demonstrated an F1 improvement**.

## Evaluation and training

All RoadSens recordings are TRAIN-only in this integration. There is no
RoadSens VAL/TEST split or claimed held-out RoadSens score. The release does
not provide reliable independent route/session grouping for a new benchmark;
random overlapping windows would be inappropriate. Existing Kaggle/LiRA
VAL/TEST records and labels are preserved exactly. This enables a controlled
comparison of cross-dataset transfer from adding RoadSens training data.

Train the existing two-head model on all real sources:

```bash
.venv/bin/python -m road_training.train_multitask \
  --data-root road_training/data_with_roadsens \
  --source real --val-source real \
  --lr 1e-4 --batch-size 256 --device cuda --precision bf16 --tensorboard
```

Use `--source both` to include synthetic data. For a RoadSens-only binary
experiment, use `road_training.train --target defect --real-dataset roadsens
--val-real-dataset kaggle` with this data root. The joint trainer requires
some TRAIN roughness labels, so use all real data for joint training.

The default supervised loader samples sliding windows uniformly; long
recordings and densely labeled sources contribute more updates. A performance
study should compare matched update budgets and report Kaggle detection and
LiRA roughness separately. Full training was not launched during integration.

## Reproduction and provenance

```bash
.venv/bin/python -m road_training.tools.prepare_roadsens \
  --base-root road_training/data_transfer_v2_mount \
  --archive 30341143.zip --output road_training/data_with_roadsens
```

The command requires a new output path, pandas and `bsdtar` (libarchive). It
reuses the local archive when present; otherwise it downloads the two sensor
archives (the second is required to recover session 99). The MD5 is checked against pinned Figshare v3, and
input/output SHA256 hashes are recorded. The GIS archive, repeated isolated
event subsets and externally hosted videos are excluded. Location, weather,
orientation and magnetometer columns never enter model inputs.

See `data/roadsens4m/source_protocol.json`, the new dataset's
`roadsens_audit.json` and `manifest.json`, and
`reports/roadsens4m_integration_20260917/` for the local audit and verification.

Data: [RoadSens-4M v3, CC BY 4.0](https://doi.org/10.6084/m9.figshare.30341143.v3).
Citation: Khandakar et al., [RoadSens-4M: A Multimodal Smartphone & Camera
Dataset for Holistic Road-way Analysis, Scientific Data 13, 1036
(2026)](https://doi.org/10.1038/s41597-026-07072-y).
