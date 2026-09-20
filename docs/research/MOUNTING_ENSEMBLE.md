> Historical research note from the source workspace. Paths, commands, and run results may refer to earlier experiments. See [the current training guide](../TRAINING.md) and [migration notes](../MIGRATION.md) before running a command.

# Ensemble trained with aligned RoadSens and mounting augmentation

Completed experiment: `reports/mounting_ensemble_20260919/`.
The four members use seeds 52, 53, 54 and 55. Seed 52 reuses the completed
matched-seed run; the other three were trained fresh. All use aligned RoadSens,
Kaggle and LiRA, instance normalization, yaw +/-20 degrees and tilt +/-10 degrees
on 75% of training windows, and the same separate IRI-head refinement.

`ensemble.json` contains the exact four checkpoint paths and SHA256 hashes.
It is an inference receipt, not a single PyTorch checkpoint. The loader verifies
those hashes and averages the four disturbance probabilities and four IRI
predictions with equal weights.

```python
import torch
from road_training.streaming_evaluation import load_teachers

model = load_teachers('reports/mounting_ensemble_20260919/ensemble.json',
                      device='cuda').eval()

# x and mask: [B,1024,7], float32 raw SI values and boolean availability.
# Phone coordinates must already be converted into the model's nominal frame.
# Acceleration includes gravity; missing values have a false mask and zero value.
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    output = model(x.cuda(), mask.cuda())

probability = output['disturbance_logit'].sigmoid()  # [B,64]
disturbance = probability >= .5
iri = output['roughness']                          # [B,64], m/km
valid = output['patch_valid']                      # [B,64]
```

The model performs instance normalization internally. Do not add global
normalization or random mounting augmentation in the app. Each window covers
10.24 seconds at 100 Hz, with 64 patches of 160 ms. The four-member parameter
count is 3,366,408. The measured p95 forward time is 3.99 ms on RTX 5090/BF16 and
10.29 ms on this workstation's CPU/FP32 with four threads. These timings exclude
observation delay and I/O and do not establish smartphone performance.

## Results

These are offline patch scores at threshold 0.5, without rolling timeline
postprocessing. Binary F1 means localized disturbance versus normal; roughness
is measured separately using LiRA spatial-section IRI MAE.

| Metric | Previous ensemble | New ensemble |
|---|---:|---:|
| Clean VAL binary F1 | 0.7326 | 0.7310 |
| Clean TEST binary F1 | 0.7224 | 0.7150 |
| Clean VAL IRI MAE, m/km | 0.2803 | 0.3005 |
| Clean TEST IRI MAE, m/km | 0.3638 | 0.3780 |
| Worst TEST F1 over ten in-range mounting perturbations | 0.6844 | 0.6961 |
| Mean TEST F1 over those perturbations | 0.7103 | 0.7085 |

The new recipe improves worst-case mounting F1 but does not improve overall
clean accuracy or roughness. The original ensemble remains available at
`reports/ensemble_causal_20260917/teacher_checkpoints.json`; it was not replaced.
The previously reported timeline postprocessing scores use a different protocol
and should not be compared directly with this table.

See the [full comparison](../reports/mounting_ensemble_20260919/report.md),
[plots](../reports/mounting_ensemble_20260919/comparison.png),
[training recipe](MOUNTING.md), and [single-model study](../reports/mounting_augmentation_20260919/matched_seed52/report.md).

All models, equal weights, the threshold and the 13 clean/rotated conditions
were frozen before TEST evaluation. The `test_read: false` flag inside the
immutable ensemble receipt describes its creation before evaluation; completed
VAL/TEST evaluation is recorded separately in `complete.json` and `evaluation/`.
No test-based weight fitting, member selection or threshold tuning occurred.

## Reproduce or resume

The frozen per-seed plans and source snapshots are in `seed53/`, `seed54/` and
`seed55/`. Their initializer refuses overwrite. To resume these same plans:

```bash
.venv/bin/python reports/mounting_ensemble_20260919/train_remaining.py
.venv/bin/python -m road_training.experiments.mounting_ensemble_eval \
  --out reports/mounting_ensemble_20260919/evaluation
```

`train_remaining.py` skips verified completed joint/head checkpoints. It runs
the three new seeds sequentially on the GPU. For a new experiment, create a new
output directory and new frozen plans rather than modifying these artifacts.
