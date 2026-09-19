# Training code guide

## Current augmented instance-normalized model

Start with `instance_model.py` for the encoder/heads, `train_multitask.py` for target reduction and focal + Huber losses, `instance_loss.py` for per-dataset weighting, and `mounting_augmentation.py` for rotation/noise.

`experiments/instance_study.py` implements dataset-balanced batches, AdamW, clipping, BF16, validation selection, early stopping, epoch metrics, and resumable checkpoints. `experiments/mounting_study.py` applies the updated augmentation recipe. The current full recipe uses batch size 256, learning rate 1e-4, 1,024-sample windows, 16-sample patches, training stride 1, and fixed sample slots for Kaggle/LiRA/RoadSens. `instance_roughness.py` optionally adapts only the IRI head on LiRA after detector selection. `mounting_ensemble_eval.py` compares four frozen members and measures latency.

Research runners freeze their data and source hashes in a plan. Old plans refer to old source locations and cannot be resumed unchanged after this migration. Create fresh plans in new output folders. Many historical runners also expect the earlier experiment artifacts named by their module constants; keeping them under `experiments/` makes this dependence explicit.

For a fresh current-recipe run, initialize its parent recipe and mounting plan in one process (requires an aligned corpus with all three real datasets):

```python
from pathlib import Path
from road_training.experiments import instance_study as base
from road_training.experiments import mounting_study as mounting

base.DATA = Path('artifacts/data_roadsens_aligned').resolve()
base.OUT = Path('reports/new_recipe').resolve()
base.initialize()  # Declares the parent recipe; does not run its ablation suite.
mounting.initialize(
    'reports/mounting_seed52', data_root=base.DATA, seed=52,
    yaw_degrees=20., tilt_degrees=10., probability=.75,
    epochs=24, steps_per_epoch=256,
)
```

Then run the declared training plan:

```bash
python -m road_training.experiments.mounting_study --run --out reports/mounting_seed52
```

Run initialization only once for each output directory. Declare additional seed plans against the same parent recipe as needed. This trains the joint model; the separate frozen-detector IRI adaptation and ensemble evaluation are subsequent experiment steps, not implicit parts of this command.

## Baselines and pretraining

| Entry point | Purpose |
| --- | --- |
| `python -m road_training.train --help` | Earlier single-task supervised classification/regression |
| `python -m road_training.train_multitask --help` | Earlier supervised two-head baseline |
| `python -m road_training.pretrain --help` | DropPatch masked autoencoding or ArcTan denoising, patch reconstruction |
| `python -m road_training.pretrain_rcd --help` | Road adaptation of Time-RCD pretraining |
| `experiments/pretraining_study/` | Scratch vs fine-tuning, frozen linear probes, reduced encoder LR |
| `experiments/rcd_study/` | RCD downstream comparison |
| `experiments/overfit/`, `real_mixup.py` | Regularization, sampling, augmentation, and MixUp ablations |
| `experiments/streaming_study.py` | Supervised/distilled causal students |

Example baseline (this uses the baseline's TRAIN normalization, not the current instance recipe):

```bash
python -m road_training.train_multitask \
  --data-root artifacts/data_roadsens_aligned --source real --val-source real \
  --lr 1e-4 --batch-size 256 --stride 1 --device cuda --precision bf16 \
  --tensorboard --output reports/baseline
```

Example reconstruction pretraining on a corpus that includes synthetic records:

```bash
python -m road_training.pretrain --method droppatch \
  --data-root artifacts/mixed_corpus --source both --epochs 12 \
  --output reports/droppatch
```

Replace `droppatch` with `arctan` for that implementation. Pretraining uses its own normalization/masking rules and does not accept the current instance-normalized supervised model interchangeably. Historical results and method-specific details are retained under `docs/research/`; they are not new performance measurements for this checkout.

## Inference and scoring

`checkpoints.load_teachers` reconstructs the saved instance model configuration, checks checkpoint hashes, and averages member probabilities/IRI. `RoadTimelineStream` provides rolling prediction updates; `timeline.py` contains overlap fusion, finalization and hysteresis. `live_inference.py` is the alternative causal-model wrapper.

`streaming_evaluation.py` implements patch, event and section metrics. Patch disturbance F1, event F1, and section IRI errors measure different things. The exported quality grades use thresholds at 2/4/6 m/km; they are derived from IRI predictions, not a separate defect-type head. The optional `alert_filter.py`/`tuned_alert_filter.py` and their experiments are additional post-processing studies; their presence does not automatically enable a filter in the default timeline configuration.
