> Historical research note from the source workspace. Paths, commands, and run results may refer to earlier experiments. See [the current training guide](../TRAINING.md) and [migration notes](../MIGRATION.md) before running a command.

# Road-sensor training workspace

Start with **`dataset.py`**. It is a plain PyTorch `Dataset`, using only NumPy
and PyTorch. Your model, pretraining and fine-tuning scripts can live here.
The prepared files are ordinary arrays: no simulator, network calls or runtime
data conversion. The original `data/` directory is self-contained; newer
dataset variants can link existing recordings to avoid copying large files.

**RoadSens-4M:** [ROADSENS.md](ROADSENS.md) describes the verified IMU-only
addition in `data_with_roadsens/`, including conservative labels, duplicates,
source filters and the unchanged Kaggle/LiRA validation and test splits.

**All-real augmentation study:** `real_mixup.py` compares sensor augmentation,
input MixUp and latent MixUp on Kaggle + LiRA + RoadSens, with matched seeds.
See the [literature review and loss details](../reports/mixup_time_series_research_20260917/report.md).
`mixup.py` keeps unknown labels masked and leaves LiRA IRI examples unmixed.
The study freezes its plan with `python -m road_training.experiments.real_mixup --init`;
`python -m road_training.experiments.real_mixup --run` trains/resumes it and evaluates the
selected checkpoints. Live epoch metrics are under
`reports/real_roadsens_mixup_20260917/*/tensorboard/` and `history.json`.

**Instance normalization:** [INSTANCE_NORMALIZATION.md](INSTANCE_NORMALIZATION.md)
documents the new per-window model and the all-real `instance_study.py` trainer.
It uses no global normalization, keeps RoadSens in training, and selects
checkpoints by validation disturbance F1 while tracking roughness separately.

**Pretraining:** [PRETRAINING.md](PRETRAINING.md) documents DropPatch and ArcTan
Diffusion, both using patch reconstruction, with a simple CUDA BF16 trainer.

**`patchtst.py`** supplies a small PatchTST-style Transformer, a masked-patch
pretrainer, and a classification/regression head. See [MODEL.md](MODEL.md) for
architecture, training examples and checkpoint reuse. `model_example.py` runs
a few pretraining steps on the selected TRAIN source.

## Joint roughness and disturbance training

`PatchTSTRoadModel` uses **one shared encoder and two independent heads at
every patch**. `train_multitask.py` trains them together from scratch:

```bash
# Real supervision: Kaggle disturbance + LiRA measured roughness.
.venv/bin/python -m road_training.train_multitask --source real --tensorboard

# Include synthetic training data; evaluate on real validation data.
.venv/bin/python -m road_training.train_multitask --source both --val-source real --tensorboard
```

Defaults: CUDA BF16, learning rate `1e-4`, batch size 256, TRAIN stride 1,
20 epochs, 1,024-sample windows and 16-sample patches. VAL windows do not
overlap. For `[B,1024,7]` input, both outputs have shape **`[B,64]`**:

```python
from road_training.dataset import RoadDataset
from road_training.patchtst import PatchTST, PatchTSTRoadModel

data = RoadDataset(source="both", return_labels=True)
model = PatchTSTRoadModel(PatchTST(train_stats=data.train_stats))
item = data[0]
output = model(item["x"][None], item["mask"][None])
roughness_m_per_km = output["roughness"]
disturbance_probability = output["disturbance_logit"].sigmoid()
valid_patches = output["patch_valid"]
```

The roughness head predicts nonnegative **overall section IRI in m/km**:
measured P79 IRI for LiRA, and geometry-derived IRI including injected defects
for synthetic data. The disturbance head predicts presence of any
real manhole, depression, bump or crack, or synthetic pothole, speed bump or
crack. Neither head pools away the patch dimension. Both have access to
bidirectional context from the input window.

The loss is `roughness_weight * Huber + disturbance_weight * binary_focal`.
Both weights default to 1. Huber has beta 1 m/km; focal gamma is 2 and uses
balanced class weights from non-overlapping TRAIN patch counts. Each loss
is averaged over its **own** valid patches. Missing roughness is never a
zero-roughness target, and a head without labels in a batch receives no
gradient. The disturbance target merges types before the existing majority
vote; ties and unknowns are ignored. A roughness patch must be fully labeled
and contained in one section; boundary/missing-label patches are ignored.

The 67 synthetic records now have `overall_iri.npy` and
`roughness_section.npy`, recovered from the original hash-verified
`road_segments.csv`. These are geometry-derived targets on nominal wheel
tracks, not acceleration-derived labels or exact driven wheel-path IRI.
Every patch estimates the IRI of its **containing road section**; the
reference does not measure IRI independently over each 0.16-second patch.
`iri.npy` and `quality_grade` still retain their original background-only
meaning. To prepare overall targets after rebuilding the dataset:

```bash
.venv/bin/python road_training/tools/prepare_overall_roughness.py --data-root road_training/data
```

| Dataset | Roughness supervision | Localized disturbance supervision |
|---|---|---|
| Kaggle | Unknown | Manhole, depression, bump and crack combined |
| LiRA-CD | Independent P79 laser-survey IRI, averaged over 100-m sections | Unknown |
| Synthetic | Full-profile geometry-derived section IRI | Injected potholes, bumps and cracks combined |

`--source real` now trains both tasks using complementary Kaggle and LiRA
labels. `--val-source real` evaluates measured LiRA roughness and Kaggle
disturbance separately. It does not demonstrate both tasks on the same drive.
Use `--real-dataset kaggle` or `--real-dataset lira` to restrict real data;
the default is `all`. `--val-real-dataset` defaults to the training filter.
LiRA-only joint training supervises only the roughness head. Kaggle-only joint
training reports missing roughness labels; use `train.py` for that single task.
Synthetic VAL covers only good/medium under the existing project cutpoints;
it cannot establish performance on bad/terrible roads.

Each epoch logs total loss, roughness Huber/MAE/RMSE and disturbance
focal/precision/recall/F1. TensorBoard includes `real/`, `synthetic/`, and
`dataset/kaggle/`, `dataset/lira/`, `dataset/synthetic/` tags;
`history.json` stores `by_source` and `by_dataset` results. The best checkpoint minimizes VAL
loss with the fixed task weights (only available tasks contribute). These
remain patch metrics, not event precision or false alerts per kilometre.
The current joint TRAIN pool has **1,633,473** supervised windows at stride 1;
VAL has **1,424** windows: 51 Kaggle, 1,303 LiRA and 70 synthetic.
Real-only TRAIN has **1,369,692** supervised windows and VAL has **1,354**.
Entirely unlabeled windows are skipped. LiRA is the largest source; simple
concatenation does not balance datasets or make repeated road passes independent.
No TEST data is loaded by either trainer.

For long runs, `progress.json` reports the current epoch, phase, batch count
and estimated remaining phase time. Loss/F1/MAE/RMSE are still recorded once
per completed epoch. Use `--num-workers 8` to overlap dataset loading with GPU
work when CPU and memory resources permit.

After selecting a checkpoint on validation, evaluate it with frozen training
normalization and a fixed disturbance probability threshold of 0.5:

```bash
.venv/bin/python -m road_training.evaluate_multitask \
  --checkpoint path/to/best.pt --source real --split test \
  --output path/to/test_metrics.json
```

The evaluator checks that the dataset manifest matches the checkpoint and
uses non-overlapping windows. It reports each dataset separately. `--split
train` can score the frozen checkpoint on non-overlapping training windows;
those metrics differ from TRAIN metrics collected during optimization.

```python
import torch
from road_training.patchtst import PatchTST, PatchTSTRoadModel

checkpoint = torch.load("path/to/best.pt", map_location="cpu", weights_only=False)
model = PatchTSTRoadModel(PatchTST(**checkpoint["config"]["encoder_config"]))
model.load_state_dict(checkpoint["model_state"])  # Also restores fixed TRAIN z-scores.
model.eval()
```

## Supervised training first

For **localized disturbance detection**, combine every annotated defect into
one positive class: real manhole, depression, bump and crack; synthetic
pothole, speed bump and crack. From the repository root:

```bash
.venv/bin/python -m road_training.train --source real --target localized_disturbance --lr 1e-4 --batch-size 256 --tensorboard
```

Choose `--source synthetic` or `--source both` to change the training data;
add `--val-source real` when training on synthetic data to evaluate real VAL.
This target is a named copy of `defect_assumed_normal`, available without
rebuilding data. Its labels are **0 no localized disturbance, 1 disturbance,
-100 unknown**. Unannotated parts of the annotated Kaggle drive are weak
negatives; wholly unlabeled drives stay unknown. Rough background alone does
not make the synthetic disturbance target positive. Synthetic targets retain
the existing geometric-contact plus response-context convention.

The types are combined **before patch voting**. A patch with seven normal,
five manhole and four crack samples therefore gets disturbance label 1.
Overlapping defect types also agree on presence, even when `kaggle_type` is
unknown. The original type tensors and annotation metadata remain available
for per-type detection audits. With default patching, labels are `[B,64]`
and logits are `[B,64,2]`. Positive-class F1 is
`per_class.disturbance.f1`, also logged as `f1/localized_disturbance` in
TensorBoard. This single-task trainer remains a patch classifier with majority
voting; short events can still lose to background. Use `train_multitask.py`
above for the joint overall-roughness head. Neither trainer scores events.

**`train.py` trains the encoder and classification head from scratch.** From
this directory, train **normal road plus Kaggle's four defect types**:

```bash
python train.py --source real --target kaggle_type --epochs 20
```

From the repository root with the existing environment:

```bash
.venv/bin/python -m road_training.train --source real --target kaggle_type --epochs 20
```

Other supervised targets and sources use the same trainer:

```bash
# Binary detection using both domains; real unannotated intervals are weak negatives.
python train.py --source both --target defect_assumed_normal --epochs 20

# Synthetic training, real validation, with the synthetic TRAIN normalizer kept fixed.
python train.py --source synthetic --val-source real --target defect_assumed_normal

# Synthetic background-quality classification.
python train.py --source synthetic --target quality_grade
```

`--source` accepts `real`, `synthetic`, or `both`; `--val-source` defaults to the
training source. A target must have usable labels in both selected splits.
`kaggle_type` has supervision only in real data; `synthetic_type` and
`quality_grade` only in synthetic data. Selecting `both` does not invent labels
for the other domain or align the two defect taxonomies. Entirely unlabeled
windows are skipped. Unknown labels inside retained windows remain ignored.

Defaults are 20 epochs, batch size 16, 1,024-sample windows, **TRAIN stride 1
sample (10 ms)**, 16-sample patches,
AdamW with learning rate `3e-4` and weight decay `0.01`, and gradient clipping
at 1.0. Training defaults to **CUDA with bfloat16 mixed precision**
(`--device cuda --precision bf16`), and validation uses the same precision.
CUDA and BF16 support are checked before loading data; there is no automatic
CPU fallback. Linear/attention activations use BF16 where supported; model
weights, normalization statistics, optimizer state and focal-loss arithmetic
stay FP32. BF16 does not need a gradient scaler. For a full-precision GPU run,
use `--precision fp32`; for CPU debugging, use `--device cpu --precision fp32`.
Normalization
uses the selected source's existing **TRAIN-only** statistics inside PatchTST;
pass raw inputs. The whole encoder is trainable, with no pretraining or freezing.
Use `python train.py --help` for the small set of options.

The trainer uses **one majority-voted class per patch** and class-weighted
softmax focal loss on each valid patch:

```text
loss(patch) = -alpha[label] * (1 - p[label])^gamma * log(p[label])
```

Defaults: `--focal-gamma 2 --focal-alpha balanced`. The probability comes from
the target's softmax (two classes for localized disturbance, five for Kaggle type).
The focusing factor reduces easy samples' contributions;
class weights increase rare classes' contributions. `alpha[c] = N / (K * n[c])`,
where counts come from valid **TRAIN majority-patch targets**, including window
overlap; `K` is the number of classes present in TRAIN. This gives mean training
weight 1. Classes absent from TRAIN get neutral weight 1. The same saved weights
and gamma are used for validation; VAL/TEST labels never fit the weights.
Unknown labels and times without observed inputs do not vote. The class with
the unique highest vote count wins, including normal. A tie or no valid votes
makes the patch unknown (`-100`). One valid vote is sufficient; there is no
minimum coverage threshold. Loss is averaged over valid target patches.
Use `--focal-alpha none` for unweighted
focal loss; adding `--focal-gamma 0` recovers ordinary cross-entropy. The loss
configuration and actual weights are saved with each checkpoint.

`majority_patch_targets()` in `train.py` groups the original timestamp labels,
then votes to produce `[batch, patches]` targets. The patch head returns logits
`[batch, patches, classes]` directly; it does not average per-sample predictions.
With 16-sample patches, the label now describes a 0.16-second interval. A short
minority defect can lose to normal within its patch; the original labels remain
available in the dataset for inspecting these cases.

TRAIN windows advance by **one sample** by default,
so adjacent 1,024-sample windows share 1,023 samples. Use `--stride 512` for
50% overlap or `--stride 1024` for no overlap. VAL always uses non-overlapping complete
windows so no sample is scored twice. Short recording tails are dropped.
These are **patch-level**, duration-weighted metrics, not event-level scores.
They use majority targets and are not directly comparable to the earlier
per-sample F1 values. No model-selection setting uses TEST.
`kaggle_type` now includes **0 normal, 1 manhole, 2 depression, 3 bump, 4 crack**.
Both this target and `defect_assumed_normal` assume normal outside annotations
on the annotated real drive; wholly unlabeled drives stay unknown. Conflicting
types, unrecognized annotations and invalid sensor intervals remain ignored.
Strict `defect` has no explicit normal samples in real VAL, so it cannot establish
binary detection performance there. Quality grade is nominal classification,
not ordinal regression; synthetic VAL currently contains only good and medium.

Each completed epoch reports **TRAIN and VAL loss, macro F1 and each class's F1**
in the terminal, plus binary defect F1 for the five-class task. `history.json`
also records loss averaged over valid patches, accuracy, precision, recall, class support,
and the confusion matrix for TRAIN and VAL. Macro F1 includes every class in
the target taxonomy; undefined class scores are zero and missing classes are
reported. For binary targets, positive-defect F1 is `per_class.defect.f1`,
which differs from the macro average. For five-class `kaggle_type`, metrics
also include `binary_defect`: collapse class 0 to normal and classes 1–4 to
defect using the same predictions. Its `per_class.defect.f1` measures detection
even when the defect type is wrong. Selection still uses **five-class macro F1**.
TRAIN metrics are collected during
updates with dropout active; VAL uses evaluation mode without gradients.
`labeled_patches` counts scored patches, and `metric_unit` is `patch`. TRAIN
class counts and focal weights are recomputed after voting, not copied from
the earlier per-sample targets. Windows with no valid voted patches are skipped.

Each run writes a new `checkpoints/<target>_<timestamp>/` directory:

- `best.pt`: full model weights from the highest **validation macro F1** epoch,
  including its fixed normalizer, configuration and validation metrics.
- `last.pt`: full model weights and metrics from the final epoch.
- `config.json`: arguments, class order/patch counts, architecture, TRAIN statistics
  and the prepared manifest's hash, plus focal-loss gamma, class weights and
  the majority-voting policy.
- `history.json`: per-epoch metrics and elapsed time.

Use `--output /path/to/new_run` to choose the directory; existing directories
are not overwritten. These are model checkpoints for inference, not exact
optimizer/RNG resume snapshots. **TEST is never loaded by the trainer** or
used for checkpoint selection. This is an offline, bidirectional baseline.
See [MODEL.md](MODEL.md#load-a-supervised-checkpoint) to load the saved model.

### Live curves per epoch

TensorBoard is optional; the dataset and terminal reporting still need only
NumPy and PyTorch. From this directory:

```bash
python -m pip install -r requirements-monitoring.txt
python train.py --source real --target kaggle_type --tensorboard
```

In another terminal, start the viewer and open http://localhost:6006:

```bash
python -m tensorboard.main --logdir checkpoints --host 127.0.0.1 --port 6006 --reload_interval 5
```

From the repository root, use `.venv/bin/python -m road_training.train` for
training and `.venv/bin/python -m tensorboard.main --logdir road_training/checkpoints`
for the viewer (with the same host/port options). Select a run's `train` and
`val` series to overlay them. The horizontal axis is **epoch**, not batch:

- `loss/focal`: mean class-weighted focal loss over valid patches.
- `f1/macro`: macro F1 over the complete target taxonomy (five classes for Kaggle).
- `f1/binary_defect`: positive-defect F1, distinct from type macro F1.
- `f1_per_class/*`: F1 for each individual class, including normal.

The trainer flushes one point per split after both passes finish each epoch,
so the viewer updates while the next epoch trains. There are no partial-epoch
or batch-averaged F1 points. Event files live in each run's `tensorboard/train/`
and `tensorboard/val/` folders and remain available after training ends.
The first point appears only after the first full epoch. With stride 1, that
means completing all 10,059 training batches and validation first. Epoch
logging does not add extra validation passes or use TEST.

## Plot labeled input windows

```bash
# Matplotlib is only needed for plotting, not for loading or training.
python -m pip install matplotlib
python plot_samples.py --split train
```

`plots/kaggle_five_class_train/` contains raw and TRAIN-z-scored PNG/PDF
figures, the exact five windows in `windows.npz`, and their recording IDs,
offsets and class counts in `selection.json`. Each column shows one actual
10.24-second dataset window: acceleration, gyroscope, speed, and a categorical
label strip. The plots use labels to select one example per class, without
ranking acceleration/gyro amplitude. Normal examples prefer moving cars
(median observed speed ≥2 m/s); defect examples prefer complete events with
normal context. These are illustrative examples, not a class-balanced sample
of the dataset. `--split val` or `--split test` plots those splits explicitly.

## Load a batch

From this directory:

```python
from torch.utils.data import DataLoader
from dataset import RoadDataset

train = RoadDataset(
    source="both",       # "real", "synthetic", or "both"
    split="train",       # "train", "val", or "test"
    window_size=1024,     # 10.24 seconds at 100 Hz
    stride=512,
)
loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
batch = next(iter(loader))
x = batch["x"]           # float32 [batch, time, 7]
mask = batch["mask"]     # bool    [batch, time, 7]
```

From the repository root, use `from road_training import RoadDataset` instead.
The default data path is relative to `dataset.py`, not your working directory.
You can move the whole directory and use it independently.

| Channels, in order | Units |
|---|---|
| accel_x, accel_y, accel_z | m/s², **including gravity**, recorded sensor axes |
| gyro_x, gyro_y, gyro_z | rad/s, recorded sensor axes |
| speed | m/s, observed speed |

`x` is **unscaled**. Invalid/stale values are zero with `mask=False`; exclude
these values from reconstruction loss. `time` contains float64 timestamps, and
`recording_id`, `source`, `dataset` and `start` identify the window. These are metadata,
not additional model inputs. Windows never cross recordings or splits. Only
complete windows are returned; short tails are dropped. `stride=None` means
non-overlapping windows. `source="both"` combines all windows; it does **not**
force a 50/50 sampling balance between domains. `dataset` is `kaggle`, `lira`,
or `synthetic`. For example:

```python
kaggle = RoadDataset(source="real", real_dataset="kaggle")
lira = RoadDataset(source="real", real_dataset="lira", return_labels=True)
all_real = RoadDataset(source="real")
synthetic = RoadDataset(source="synthetic")
mixed = RoadDataset(source="both")
```

## Optional normalization and patches

Statistics are already fitted using valid **training** observations separately
for real, synthetic and both. The selected source and `real_dataset` filter
determine `train_stats`. Kaggle-filtered statistics reproduce the original
pre-LiRA statistics exactly. LiRA-only gyroscopes have zero observations and
neutral mean 0/std 1; their values remain masked throughout the model.
Reuse the **same statistics from your training dataset** when evaluating on a
different source; do not switch scalers when switching the evaluation filter.

```python
import torch

mean = torch.tensor(train.train_stats["mean"], dtype=torch.float32)
std = torch.tensor(train.train_stats["std"], dtype=torch.float32)
x = ((batch["x"] - mean) / std).masked_fill(~mask, 0)

# Patching stays in your training/model code. Each patch keeps all 7 channels.
patches = x.reshape(x.shape[0], -1, 16, 7)
patch_mask = mask.reshape_as(patches)
```

For next-patch prediction, load `(context_patches + 1) * patch_size` samples.
Use `patches[:, :-1]` as inputs, `patches[:, 1:]` as targets and
`patch_mask[:, 1:]` to mask the loss. `example.py` demonstrates this with 64
input patches and 16 samples per patch. **This shifted objective requires a
causal model.** The supplied PatchTST encoder is bidirectional and uses masked
patch reconstruction through `PatchTSTPretrainer`; do not train it on that
shifted objective. Its normalization and patching are already inside the
encoder: pass raw dataset inputs without applying the above normalization twice.

```bash
python example.py --source both
python example.py --source real
python example.py --source synthetic
```

## Included data

| Source | Train | Validation | Test |
|---|---|---|---|
| Kaggle real IMU + speed | 3 recordings; 107.27 min | 1 segment; 8.78 min | 1 segment; 8.91 min |
| Corrected synthetic IMU + speed | 53 recordings; 53 min | 14 recordings; 14 min | None |
| LiRA acceleration + speed | 7 passes of CPH1; 6.384 h | 13 passes of M3; 4.256 h | 8 passes of M13; 1.266 h |

The training pool totals **63 recordings / about 9.055 hours**. There are 100
recordings or split segments across the whole package. Full counts and source
hashes are in `data/manifest.json`.

- **Kaggle:** Larisa's existing purged chronological train/validation/test
  segments, plus the two wholly unlabeled Thessaloniki drives in training only.
  The approximately 100 Hz IMU is interpolated onto an exact 100 Hz grid inside
  each split. Gaps longer than 50 ms are masked. Speed uses the last observation
  and is masked after 3 seconds without an update. No held-out sample is used
  to interpolate a training sample or fit statistics. The loader is not itself
  a causal preprocessing pipeline: this offline IMU interpolation uses the two
  bracketing observations within the same split.
- **Synthetic:** all 63 unique corrected-observer physical recordings, plus
  the four fresh general-road recordings produced for the representative
  gallery. Those add isolated/clustered potholes, bumps and all four roughness
  grades. Sensor values, masks and clocks are copied exactly. Related physical
  variants retain their original family split. Experimental alias/repeat views,
  old buggy observation exports and pending steering candidates are excluded.
- **LiRA-CD:** raw `acc.xyz` (approximately 50 Hz, g) converted to m/s²;
  speed converted from km/h to m/s. The most recent observations are held
  forward on the 100 Hz grid, without future interpolation. Acceleration older
  than 0.1 s and speed older than 0.25 s are masked. This adds no sensor bandwidth.
  All three gyroscopes are missing and masked. GPS aligns independent targets
  only; it never enters the encoder. Entire roads, including all devices and
  repeated passes, stay in their fixed splits. CPH6 remains excluded because
  its measured geometry was previously used for simulator demonstrations.

There is no held-out synthetic test set in this corrected corpus.
`source="synthetic", split="test"` raises a clear error. With `source="both"`,
the test split contains Kaggle and LiRA test data. Real validation/test remain
separate from pretraining, including unsupervised pretraining.

### LiRA reference alignment

The original P79 CSV supplies left/right IRI at 10-m stations. Each target is
the mean of ten rows and both wheel tracks over a 100-m section. Reference
GPS comes from the original same-road elevation CSV joined by published
distance; embedded HDF5 reference tables are not assumed to match those CSVs.
GPS observations may bracket a sensor time **for offline label alignment only**.
`sensor_source_time.npy` records the acquisition times of held acceleration
and speed, allowing the absence of future sensor values to be audited.

Sections require at least 90 m coverage, two seconds of sensor data, 95%
continuity, mean speed at least 5 m/s and tenth-percentile speed at least 3 m/s.
GPS intervals over 3 s, reference distances over 20 m and implausible jumps
are excluded. Passes missing required sensors or with less than 90% sensor
coverage are excluded. An unsupported trailing IRI row in M13 is dropped;
GPS positions are never extrapolated to invent reference coverage.

The admitted targets cover **2.708 h TRAIN, 3.519 h VAL and 0.952 h TEST**.
There are 1,222 / 2,833 / 861 section traversals respectively, representing
235 / 258 / 119 distinct labeled sections on only **three roads**.
Unaligned samples can serve unsupervised learning; their roughness labels
remain unknown. No disturbance positives or negatives are inferred from IRI.
`quality_grade` also remains unknown for LiRA: this import supplies measured
regression targets, not an invented official quality taxonomy.

GPS uncertainty, different survey dates/wheel paths and repeated passes limit
evaluation. Current roughness metrics weight patches by time, not unique road
sections. Section-level aggregation and broader road coverage are still needed
before making deployment or benchmark-comparability claims. M13 was prepared
and checked for data integrity during integration; no model was trained or
evaluated on it. See `reports/lira_integration/` in the parent repository for
the import audit and small TRAIN/VAL CUDA BF16 integration runs.

LiRA-CD attribution: Skar et al. (2023), Technical University of Denmark,
[Live Road Assessment Custom Dataset](https://data.dtu.dk/collections/Live_Road_Assessment_Custom_Dataset_LiRA-CD_/6659909),
CC BY 4.0. Source hashes and the conversion recipe are saved in the manifest.

## Optional labels for fine-tuning

```python
data = RoadDataset(source="real", split="train", return_labels=True)
item = data[0]
target = item["labels"]["kaggle_type"]  # int64 [time]; -100 means unknown
```

The dataset returns a single label tensor **`[1024]`** or a batch **`[16,1024]`**,
with dtype **`torch.int64`**. The trainer first groups this as `[16,64,16]`,
then majority-votes to **`[16,64]`**: 16 windows, 64 patch labels per window.
These are integer class IDs, not one-hot vectors.

```python
from train import majority_patch_targets

labels, valid = majority_patch_targets(batch, "kaggle_type", patch_length=16)
# labels: [16, 64]; valid: [16, 64]
```

The default patch head returns **`[16,64,5]`**, five logits per patch. Focal loss
has shape `[16,64]` before reduction; ignored patches contribute zero. F1
counts valid patches. Training window stride remains 1, and VAL windows remain
non-overlapping. `output_level="sample"` is still available for older models,
but its head weights have a different shape from the new patch head.

`examples/sample_labels.npy` contains an actual batch of 16 adjacent TRAIN
windows at stride 1. **`examples/sample_patch_labels.npy`** contains the same
labels grouped as `[16,64,16]` before voting. **`examples/sample_majority_labels.npy`**
contains the actual `[16,64]` patch targets; `sample_majority_valid.npy` contains
their validity mask. The accompanying `sample_labels.json` records
the source and offsets.
The first window's 1,024 labels are:

```text
samples    0:594  → 0 (normal)   — 594 samples
samples  594:644  → 1 (manhole)  —  50 samples
samples  644:995  → 0 (normal)   — 351 samples
samples 995:1024  → 4 (crack)    —  29 samples
```

For example, zero-based patch 37 covers samples 592–607, with two normal
labels and fourteen manhole labels. Its majority target is **`1` (manhole)**.

Stored labels remain per sample. Majority voting happens in the trainer;
the loader applies no event-centered cropping or automatic event balancing.
`DataLoader` batches the source arrays normally. Classification
targets use `-100`, compatible with `CrossEntropyLoss(ignore_index=-100)`.
For a binary loss, explicitly mask targets equal to `-100`.

| Label | Meaning |
|---|---|
| `localized_disturbance` | 0 no localized disturbance, 1 any defect, -100 unknown. Derived copy of `defect_assumed_normal`; includes real manhole/depression/bump/crack and synthetic pothole/speed bump/crack, including overlapping types. Original type labels are retained. |
| `defect` | 0 normal, 1 defect, -100 unknown. Real unannotated intervals stay unknown. Synthetic uses the original generic context target and validity mask. |
| `defect_assumed_normal` | Explicit weak-label alternative: unannotated intervals of the **annotated** Kaggle drive are assumed normal. The wholly unlabeled drives stay unknown. Equals `defect` for synthetic data. |
| `kaggle_type` | **0 normal, 1 manhole, 2 depression, 3 bump, 4 crack**. Unannotated intervals of the annotated Kaggle drive are assumed normal. Wholly unlabeled drives, conflicting/unknown types, invalid inputs and all synthetic samples stay -100. |
| `synthetic_type` | 0 pothole, 1 speed_bump, 2 crack; unknown without a single unambiguous active kind. |
| `quality_grade` | Synthetic background roughness: 0 good, 1 medium, 2 bad, 3 terrible. Unknown for Kaggle and LiRA. |
| `iri`, `iri_valid` | Synthetic background IRI in m/km and its validity mask. Invalid returned values are zero. |
| `overall_iri`, `overall_iri_valid` | LiRA measured P79 section IRI or synthetic geometry-derived section IRI including injected defects, in m/km. Kaggle and unaligned LiRA samples are unknown. Invalid returned values are zero, with validity false. |
| `roughness_section` | Recording-local section ID for overall IRI; -1 means unknown. Used only to mask supervision across boundaries, never as a model input. |

Kaggle depression is **not automatically mapped to synthetic pothole**. Kaggle
event severity is not used as a road-roughness grade. Background IRI excludes
injected defects; it is not a total-road severity score. Synthetic generic
context, geometric contact and the original training event-support convention
are different targets. Each synthetic recording retains `original_targets.npz`
and full `metadata.json` for specialized event/severity/quality fine-tuning.
Kaggle metadata retains all applicable original annotation fields.

Prepared manifest **version 2** uses this five-class mapping in `labels.npy`.
The loader rejects old four-class labels when `return_labels=True`; rebuild
old exports with the updated preparation script. Old four-class checkpoints
retain their original class order and cannot serve as five-class classifiers.

## Small directory layout

```text
dataset.py               # The only loader your training script needs
patchtst.py              # Shared Transformer encoder + pretraining/prediction heads
train.py                 # Supervised classification from scratch; TRAIN/VAL only
train_multitask.py       # Two patch heads: overall IRI + localized disturbance
evaluate_multitask.py    # Frozen checkpoint, non-overlapping windows, per-dataset metrics
plot_samples.py          # Real sensor windows and their five-class label strips
model_example.py         # A few actual masked-pretraining updates on TRAIN data
MODEL.md                 # Architecture, losses, fine-tuning and checkpoints
example.py               # DataLoader + optional normalization + next-patch shapes
requirements.txt         # NumPy and PyTorch
requirements-prepare.txt # Optional h5py/pandas/scipy for one-time LiRA conversion
data/manifest.json       # Sources, splits, train statistics, labels, checksums
data/records/<id>/        # x, mask, time, labels, iri as memory-mappable .npy files
tools/prepare_data.py    # One-time conversion; never imported by the Dataset
tools/prepare_overall_roughness.py  # Copy verified full-profile section IRI targets
tools/prepare_lira.py    # Import raw LiRA sensors and independent P79 IRI
tests/test_dataset.py    # Boundaries, masks, statistics, multi-worker loading
tests/test_patchtst.py   # Hidden-target isolation, missing values, gradients, checkpoints
tests/test_train.py      # Supervised masking, aggregate F1, validation, updates
tests/test_multitask.py  # Separate masks, head gradients, section boundaries, BF16
tests/test_lira.py       # Past-only sensor timing, SI units, reference support
```

The dataset maps files lazily and caches at most eight recordings per worker.
It copies only each returned window, so in-place augmentation cannot corrupt
the underlying recording. Only the preparation script needs the old repository.
To rebuild into a **new** destination:

```bash
python tools/prepare_data.py --repository /path/to/pothole --output /path/to/new_data
python tools/prepare_overall_roughness.py --data-root /path/to/new_data
python -m pip install -r requirements-prepare.txt
python tools/prepare_lira.py --data-root /path/to/new_data --source-root /path/to/pothole/data/road_quality_reference
python -m unittest discover -s tests -v
```

The current `data/` already includes LiRA; the importer refuses duplicate
imports. `data/manifest.before_lira.json` preserves the old corpus manifest.
Earlier checkpoints retain their own saved normalizers; use `--real-dataset
kaggle` when reproducing a Kaggle-only experiment with new code.

Dependencies can be installed with `python -m pip install -r requirements.txt`.
The model and supervised trainer add no dependencies. `model_example.py` is
only a short pretraining smoke run; `train.py` is the supervised baseline.
No simulator or pre-trained weights are bundled here.
