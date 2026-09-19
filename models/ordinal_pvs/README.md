# Ordinal road quality + disturbance ensemble

Four completed ordinal + PVS models, seeds **52–55**, packaged for inference.
Every trained tensor exactly matches the selected research checkpoint. Each
file is about 3 MB; optimizer state, local recording paths and training data
are omitted. `ensemble.json` contains portable paths, package checksums,
source hashes and selected epochs. Load with `weights_only=True` through the
loader below.

This was the highest LiRA-validation-scoring **trained ordinal variant**.
Across regression and ordinal variants, regression + PVS had a slightly
higher validation F1 (.7661 versus .7581). These are the completed ordinal
study weights, not SAM/SWA or distilled replacements.

## Inputs and outputs

- Input: float32 `[B,1024,4]`, **accel_x, accel_y, accel_z, speed**, at 100 Hz.
- Units: acceleration including gravity in m/s²; speed in m/s.
- Nominal frame: x forward, y left, z up; see [orientation notes](../../docs/DATA.md).
- Supply a matching boolean observation mask. Missing values are zero/false.
- Masked per-window instance normalization and statistics for both heads are
  inside the model; do not normalize externally.
- PatchTST: width 128, three bidirectional layers, four attention heads,
  FFN width 512; 16-sample patches, 64 patches per window.
- `quality_probability`: `[B,64,3]`, ordered **good, medium, bad**.
- `quality_class`: `[B,64]`, values 0/1/2 using the ordinal median decision.
- `disturbance_probability` and `disturbance_logit`: `[B,64]`.
- `patch_valid`: `[B,64]`; ignore predictions where false.

Members' probabilities are averaged equally. The quality decision is
`P(medium or bad) >= 0.5` plus `P(bad) > 0.5`, not an argmax decision.
The disturbance target combines manholes, depressions, bumps and cracks.
It is not a dedicated pothole classifier.

LiRA reference classes use IRI below 95/63.36 m/km for good, from that lower
boundary through 170/63.36 inclusive for medium, and above the upper boundary
for bad (approximately 1.4994 and 2.6831 m/km). **The ordinal head does not
output calibrated numeric IRI.** It must not be passed to the existing
IRI-based `RoadTimelineStream`/replay exporter as an IRI regression model.
This package supplies window inference. For rolling inference, the fixed Kalman
filter, and the web replay, use the dedicated ordinal path described in
[CSV inference](../../docs/CSV_INFERENCE.md). The original regression replay
remains available through its existing interface.

## Load and infer

Run from the repository root after installing the project dependencies:

```python
import torch
from road_training.ordinal_checkpoints import load_ordinal_ensemble

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_ordinal_ensemble('models/ordinal_pvs/ensemble.json', device=device)
# samples: [B,1024,4], raw SI measurements; observed: matching boolean mask.
x = torch.as_tensor(samples, dtype=torch.float32, device=device)
mask = torch.as_tensor(observed, dtype=torch.bool, device=device)
with torch.inference_mode(), torch.autocast(
    device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'
):
    output = model(x, mask)
quality = output['quality_probability']
disturbance = output['disturbance_probability']
valid = output['patch_valid']
```

## Training and augmentation

The original encoder/detector was trained with Kaggle Greece, LiRA-CD and
aligned RoadSens using acceleration + speed. Ordinal adaptation used LiRA
IRI classes and all nine PVS recordings as auxiliary training-only data.
PVS labels are acceleration-derived proxies with separate learned
vehicle-specific calibration; they do not supply measured IRI or detector
negatives. During ordinal adaptation, the encoder, statistics embedding and
disturbance head stayed frozen; the roughness/ordinal parameters were trained.

Augmentation was used during training: mounting yaw within ±20°, tilt within
±10°, applied to 75% of eligible windows, plus small accelerometer noise
(0.005 m/s² Gaussian scale, clipped at three standard deviations) and
per-window bias within ±0.02 m/s². Identical held readings share their noise.
Speed, timestamps, masks and targets are preserved. Augmentation is applied
before instance normalization and is not applied to validation/test inputs.
No synthetic data or MIT/UMass data trained these four checkpoints.

## Recorded performance

Historical evaluations are included in [metrics.json](metrics.json).

| Metric | Validation | Test |
|---|---:|---:|
| LiRA physical-section roughness macro F1 | .7581 | .7976 |
| Kaggle binary patch disturbance F1 | .7094 | .6970 |
| Kaggle binary patch precision | .6061 | .5495 |
| Kaggle binary patch recall | .8552 | .9528 |

LiRA scores average good and medium because those held-out splits have no
bad reference sections. Section scores pool repeated traversals. These
metrics use fixed decisions and no Kalman filtering.

On MIT/UMass original released XYZ, pooled physical-section three-class
macro F1 was .7976; individual-traversal section macro F1 was .7457. The 490
pooled sections contain 213 good, 74 medium and 203 bad. These are preliminary
external results: horizontal sensor artifacts and provisional urban
laser-profile registration remain limitations. MIT/UMass has no defect
ground truth. Existing test data have prior project exposure, so none of
these numbers establishes accuracy on unseen phones or cities.

See [the ordinal study](../../docs/ORDINAL.md) for the matched comparisons
and [verification.json](verification.json) for packaging/inference checks.
