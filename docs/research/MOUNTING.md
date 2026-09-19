> Historical research note from the source workspace. Paths, commands, and run results may refer to earlier experiments. See [the current training guide](../TRAINING.md) and [migration notes](../MIGRATION.md) before running a command.

# Phone orientation and mounting augmentation

The completed four-model run, inference example, and validation/test comparison
are documented in [MOUNTING_ENSEMBLE.md](MOUNTING_ENSEMBLE.md).

The new recipe combines **nominal RoadSens frame alignment** with **random
residual mounting errors during training**. Both happen on raw SI values,
before the model's instance normalization. These are separate operations:
alignment removes the approximately 90-degree source mismatch; augmentation
teaches tolerance to smaller remaining mounting differences.

## Prepared dataset

Use `road_training/data_roadsens_aligned`. It contains both real and synthetic
recordings, with the same `RoadDataset` source filters and train/val/test splits
as `data_with_roadsens`. Only RoadSens sensor coordinates change. This is a new
directory; previous datasets and checkpoints are preserved.

```python
from road_training.dataset import RoadDataset

train = RoadDataset('road_training/data_roadsens_aligned', source='real',
                    split='train', stride=1, return_labels=True)
```

The intended model convention is right-handed: X forward, Y left, Z up. For
an upright Android phone whose screen faces rearward:

```text
model X = -device Z
model Y = -device X
model Z =  device Y
```

`tools/align_roadsens.py` applies this same proper rotation to acceleration
and gyro, permutes their missingness masks, and leaves speed unchanged.
Gravity is retained. Timing, labels, event intervals, splits and all non-RoadSens
records are unchanged. All 97 RoadSens recordings remain TRAIN-only.

The [RoadSens publication](https://www.nature.com/articles/s41597-026-07072-y)
describes a windshield mount with the rear camera viewing the road. The
recordings have positive device-Y gravity. Together these support the nominal
mapping above. Exact per-drive yaw/tilt is not documented, so the conversion
does not claim exact calibration. Kaggle's paper describes forward X and up Z;
LiRA's horizontal-axis convention remains unverified. No per-drive orientation
is inferred from labels, future samples, or held-out performance.

Rebuild into a **new** destination:

```bash
.venv/bin/python -m road_training.tools.align_roadsens \
  --base-root road_training/data_with_roadsens \
  --output road_training/data_roadsens_aligned
```

The new manifest includes the rotation and source hashes; `alignment_audit.json`
records per-record acceleration baselines before/after. Legacy global statistics
are recomputed from TRAIN for loader compatibility. The recommended trainer
continues to use instance normalization, not those global statistics.

## Training augmentation

`mounting_augmentation.py` uses these configurable defaults:

| Setting | Default |
|---|---|
| Yaw about approximate vertical | Uniform -20 to +20 degrees |
| Tilt about a random horizontal axis | Uniform -10 to +10 degrees |
| Apply a mounting rotation | 75% of windows |
| Retain original orientation | 25% of windows |
| Existing sensor noise/bias | Enabled separately |

One rotation is constant throughout the entire window and shared by acceleration
and gyro. Timestamps, task labels and speed do not change. A rotation preserves
the magnitude of each vector; the separately enabled sensor noise can change it.
It does not randomly rotate every sample, which would introduce nonphysical
motion into a fixed-mount example. It does not model picking up the phone or a
loose mount moving during the drive.

The window's mean total acceleration supplies an approximate vertical. This
supports both original Y-up records and aligned Z-up records. Driving dynamics
can bias that estimate. This is a TRAIN-only perturbation, not a live orientation
estimator. Mean magnitudes outside 5–15 m/s², absent acceleration, or any partially
observed XYZ triad disable rotation for that window. Fully missing triads
(LiRA gyroscope), missing speed, and all-channel padding remain supported.

```python
from road_training.mounting_augmentation import perturb

if model.training:
    x = perturb(x, mask, seed=72, step=global_step,
                yaw_degrees=20, tilt_degrees=10, probability=.75)
output = model(x, mask)  # normalization and patches happen inside the model
```

Geometry is computed in FP32 even under BF16 autocast. The dedicated random
generator is deterministic for seed/step and does not consume the model's RNG.

## Training runner

`mounting_study.py` reuses the successful real-only trainer with aligned data
and the new augmentation. It changes the training augmenter in a scoped adapter;
the historical trainer and its frozen source hashes remain intact. Existing
checkpoints still reflect their original <=5-degree augmentation.

```bash
.venv/bin/python -m road_training.experiments.mounting_study --init
.venv/bin/python -m road_training.experiments.mounting_study --run
```

The default run uses seed 72, CUDA/BF16, LR 1e-4, batch 256 (64 Kaggle, 128 LiRA,
64 RoadSens), 24 maximum epochs, and the previous focal/Huber loss and early
stopping. VAL is clean and unaugmented; no TEST evaluation or automatic checkpoint
promotion occurs. `--init` freezes the data/code/settings; `--run` uses that plan.
Per-epoch metrics and TensorBoard logs use the existing trainer. A subsequent
IRI-head refinement must retain or recheck mounting robustness.

Implementation tests and a two-update training smoke test validate the pipeline,
not F1 improvement. To assess robustness, compare clean validation against fixed
per-record yaw angles 0, +/-10, +/-20, +/-30 degrees and tilt errors, then freeze
the recipe before final test evaluation. Rotated copies stay in their parent
split and must not be counted as independent drives. Real phone recordings are
still needed to validate actual mounting and device transfer.

The same-window shared rotation is consistent with the IMU augmentation used in
[Deep Learning for Inertial Sensor Alignment](https://arxiv.org/html/2212.11120v2).
That paper estimates mounting angle; its results do not establish F1 improvements
for this road-condition model. Our angle ranges are initial design choices.

## Mobile app

Keep the nominal phone-to-model conversion (or a calibrated rotation for the
actual mount) at inference. Do not apply random augmentation in the app. Do not
rotate already aligned RoadSens data a second time. The new augmentation covers
residual yaw/tilt around the nominal frame; it is not a substitute for the large
portrait-to-model coordinate conversion.
