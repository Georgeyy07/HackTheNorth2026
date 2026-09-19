# Acceleration + speed ensemble

Four frozen, independently trained seeds **52–55**. The `.pt` files contain only model configuration and trained tensors, about 2.9 MB each. Optimizer state, recording paths and training data are omitted. Every tensor equals the original selected checkpoint exactly; source and packaged checkpoint SHA-256 hashes are recorded in `ensemble.json`.

## Input and architecture

- Float32 `[B,1024,4]`: **accel_x, accel_y, accel_z, speed**. Matching boolean mask; zero/false for missing inputs. No gyroscope input.
- 100 Hz, acceleration including gravity in m/s², speed in m/s. Nominal frame: x forward, y left, z up; source calibration is approximate. See [orientation notes](../../docs/DATA.md).
- Current masked per-window instance normalization, epsilon 1e-5; the same window's `asinh(mean)`, `log(std)` and observed fraction feed the statistics branch for **both** heads. No global normalization and no experimental normalization floor.
- Channel-independent PatchTST: patch length 16, 64 patches, width 128, three transformer layers, four attention heads, feed-forward width 512, dropout .1. Each member has 741,762 parameters.
- Per-patch positive IRI in m/km and binary localized-disturbance logit. Ensemble averages member sigmoid probabilities and IRI equally.

## Training provenance

Real Kaggle, LiRA and aligned RoadSens only; no synthetic training or pretraining. Batch size 256 (64/128/64 windows per dataset), AdamW learning rate 1e-4, weight decay .01, BF16 CUDA. Training candidates have stride one; 256 batches are sampled per epoch, up to 24 epochs with validation-led early stopping. Augmentation applies gravity-axis yaw ±20° and tilt ±10° to 75% of windows, plus small acceleration noise/bias. Speed and missing-input masks are preserved.

The joint loss uses TRAIN-balanced binary focal loss (gamma 2, Kaggle/RoadSens weights 1/.25) and Huber IRI loss weighted .25. Kaggle validation F1 selects the joint checkpoint. Then only the roughness head is adapted on clean LiRA TRAIN, selected by LiRA validation IRI error, for at most 12 epochs. Encoder, statistics and detector tensors remain frozen during adaptation. Four final members were frozen before the reported TEST scoring.

## Recorded measurements

These are existing September 19, 2026 experiment results, not a new evaluation from this repository migration.

| Protocol / metric | Validation | Test |
| --- | ---: | ---: |
| Offline Kaggle binary patch F1, threshold .5 | .7094 | .6970 |
| Offline LiRA section IRI MAE, m/km | .2870 | .3580 |
| Rolling Kaggle patch F1, fixed Kalman | .695 | .719 |
| Rolling Kaggle event precision, fixed Kalman | .780 | .852 |
| Rolling Kaggle event recall, fixed Kalman | .640 | .684 |
| Rolling Kaggle event F1, fixed Kalman | .703 | .759 |

Rolling inference uses a 10.24-second context, 160-ms hop, three predictions per finalized target patch weighted 1/2/3, and 320-ms finalization delay. The fixed Kalman filter acts on finalized disturbance scores with R=1, Q/R=3.2, onset=.70 and offset=.50. It leaves IRI unchanged. Provisional tail predictions remain unfiltered. Missing patches and new drives reset filtering. Covariance is not calibrated prediction uncertainty.

Filtering improves alert precision at a recall cost; it is not the highest-F1 profile in the prior comparison. Threshold-only .70/.50 obtained rolling TEST event F1 .795 versus .759 with Kalman. The earlier Kalman profile is retained here by user preference, with no new TEST tuning.

Kaggle supplies disturbance labels but no measured IRI; LiRA supplies section IRI but no disturbance labels. RoadSens is TRAIN-only. The disturbance class includes manholes, depressions, bumps and cracks; it is not a dedicated pothole classifier. Existing TEST has repeated historical exposure, and these results do not establish accuracy on unseen smartphones or cities.

## Run

```python
from road_training.checkpoints import load_teachers
from road_training.timeline_stream import RoadTimelineStream

model = load_teachers('models/acceleration_speed/ensemble.json', device='cuda')
stream = RoadTimelineStream(model)  # Default: consensus + earlier fixed Kalman.
rows = stream.push(acceleration_and_speed, observed_mask)  # [N,4], raw SI.
```

Full drive export is documented in the [repository README](../../README.md). Canonical stored datasets can still contain raw gyro for provenance; the exporter discards it before model inference.

## Integration verification

[Verification receipt](verification.json): 203 Python tests and 5 JavaScript replay tests passed. A CUDA BF16 smoke check covered all three real sources and one augmented optimizer update in memory. On the first 1,607 samples of the existing Kaggle TEST recording, 98 finalized patches matched the earlier fixed-Kalman run within 1.4e-17 in disturbance score and exactly in IRI. The remaining two whole patches plus one partial patch stayed provisional. This checks migration correctness, not new generalization performance. Checkpoints on disk were not modified by the optimizer smoke test.
