> Historical research note from the source workspace. Paths, commands, and run results may refer to earlier experiments. See [the current training guide](../TRAINING.md) and [migration notes](../MIGRATION.md) before running a command.

# PatchTST-style road-sensor model

`patchtst.py` is self-contained PyTorch code. It follows PatchTST's two central
ideas: temporal patches and a shared Transformer applied independently to each
sensor channel. See the [original paper](https://arxiv.org/abs/2211.14730) and
[official implementation](https://github.com/yuqinie98/PatchTST).

This is a road-data adaptation: pre-norm LayerNorm Transformer blocks, explicit
missing-value masks, fixed training-set z-scores, non-overlapping patches and
patch-level predictions, plus optional per-sample/window heads. It is not a reproduction of the
published forecasting experiments. It does not implement residual-attention
score accumulation or RevIN.

## Architecture and tensor shapes

Defaults: **7 channels, 16 samples per patch, 128-dimensional embeddings,
3 Transformer layers, 4 attention heads, 512-unit feed-forward layers,
GELU, dropout 0.1**. The pretrainer has **634,256 parameters**. Learned positions
support up to 256 patches; set `max_patches` if you need longer windows.

```text
Dataset x                    [B, 1024, 7]        raw sensor values
Fixed TRAIN z-score          [B, 1024, 7]
Separate non-overlap patches [B, 7, 64, 16]
Shared patch projection      [B, 7, 64, 128]
Shared Transformer           [B*7, 64, 128]      bidirectional attention
Encoded features             [B, 7, 64, 128]
```

The encoder does not concatenate all seven variates into one patch token.
Each variate is processed separately with shared weights. The downstream head
concatenates the seven **encoded** channel representations so it can learn
relationships between acceleration, angular velocity and speed.

Input length must be divisible by `patch_length`. No samples are silently
discarded or padded by the model. Patches are non-overlapping so the same
prepared sample cannot appear in both a hidden patch and a visible patch.

## Supervised training from scratch

The combined model is **`PatchTSTRoadModel`**, with two heads sharing one
encoder forward pass. For `[B,1024,7]` input:

```text
Shared PatchTST features       [B,7,64,128]
Concatenate channels per patch [B,64,896]
  roughness_head:   LN -> Linear(896,128) -> GELU -> Dropout -> Linear(128,1)
                   -> Softplus -> overall IRI [B,64], in m/km
  disturbance_head: LN -> Linear(896,128) -> GELU -> Dropout -> Linear(128,1)
                   -> presence logits [B,64]; sigmoid gives probabilities
```

There is no pooling across patches in either head. The roughness target is
the containing section's full-profile IRI, not a separate measurement for
each 16-sample patch. Both heads use the window's bidirectional context.
`train_multitask.py` uses independently masked Huber and binary focal losses.
Kaggle has disturbance labels only; LiRA has independently measured P79
section IRI only; prepared synthetic data has both targets. Missing tasks do
not contribute loss. LiRA's unavailable gyroscope channels are masked and
skipped by the shared encoder. `--source real` combines Kaggle and LiRA;
`--real-dataset kaggle|lira` restricts that selection and its TRAIN normalizer.
Per-dataset metrics keep measured LiRA IRI separate from synthetic IRI.
See [joint training](README.md#joint-roughness-and-disturbance-training) for
commands, target provenance, metric separation, and checkpoint loading.

The original single-task trainer remains available:

`train.py` initializes a new encoder and patch classifier, then trains
both jointly with class-weighted softmax focal loss (`gamma=2`). Class weights
are fitted on TRAIN majority-patch counts and reused for validation. No pretraining is required:

```bash
python train.py --source real --target kaggle_type --epochs 20
```

The trainer defaults to CUDA with BF16 autocast for both TRAIN and VAL;
weights and focal-loss arithmetic stay FP32. Add `--tensorboard` for live
loss and F1 curves updated once per completed epoch. Terminal reporting
always includes both splits. See [live monitoring instructions](README.md#live-curves-per-epoch).

It supplies the raw seven-channel inputs and observation mask to the model;
labels are used only by the loss and metrics. The classifier returns
`[B, number_of_patches, number_of_classes]`, one prediction per patch.
Targets are obtained by majority voting over known, observed timestamp labels;
ties and patches without valid votes are excluded. The default objective is
five-class Kaggle classification (normal, manhole, depression, bump, crack);
binary detection and synthetic type/quality
classification are selectable. See [README.md](README.md#supervised-training-first)
for target semantics, metrics, splits and options. The highest validation
macro F1 selects `best.pt`; TEST is not used.

Use `--target localized_disturbance` to train one combined class for all real
manholes, depressions, bumps and cracks, and synthetic potholes, speed bumps
and cracks. The head ends in `Linear(128,2)` and returns `[B,N,2]`, with class
order `no_disturbance, disturbance`. Presence labels are combined before
majority voting; original defect-type labels are retained. Positive-class F1
is `per_class.disturbance.f1`. Background roughness is a separate property;
use `PatchTSTRoadModel` and `train_multitask.py` for both outputs together.

For this default target the patch head is `LayerNorm(896) → Linear(896,128)
→ GELU → Dropout → Linear(128,5)`. Each encoded patch produces **5 logits**,
giving **`[B,N,5]`**, where `N=T/P` and `P` is the patch length. Timestamp labels
`[B,T]` are grouped as `[B,N,P]` and voted into **`[B,N]`** targets. With the
defaults, targets are **`[16,64]`** and logits are **`[16,64,5]`**.
The unique most frequent valid class wins; ties and empty votes become -100.
One known, observed sample is enough to vote when the rest are unknown.
Normal participates in voting and focal loss
alongside the four defect types. Normal outside annotations is an explicit
weak-label assumption; wholly unlabeled drives and ambiguous types stay ignored.
Five-class scores include normal. A separate binary metric collapses the four
predicted defect classes, so type mistakes can still count as correct detection.
All these metrics now score patches against voted targets, not timestamps.

### Load a supervised checkpoint

From this directory, replace the path with the one printed by the trainer:

```python
import torch
from patchtst import PatchTST, PatchTSTPredictor

saved = torch.load("checkpoints/<run>/best.pt", map_location="cpu", weights_only=True)
config = saved["config"]
encoder = PatchTST(**config["encoder_config"])
model = PatchTSTPredictor(encoder, output_dim=len(config["classes"]),
                         output_level=config.get("output_level", "sample"))
model.load_state_dict(saved["model_state"])  # Restores the head and TRAIN mean/std.
model.eval()
with torch.no_grad():
    logits = model(batch["x"], batch["mask"])  # Raw, unnormalized CPU batch.
    predicted_class = logits.argmax(-1)       # [B, patches] for new checkpoints.
```

Keep the saved training normalizer when evaluating another source. For a GPU,
move both the model and input tensors to the same device. Checkpoints contain
model weights and configuration; the simple trainer has no optimizer-resume
mode. Earlier per-sample head weights cannot load into the smaller patch head;
restore the recorded output level to inspect an old checkpoint, or initialize
a new patch head when transferring its encoder. The pretraining components
below remain available for later comparisons.

## Masked-patch pretraining

For the implemented **DropPatch** and **ArcTan Diffusion** reconstruction
methods, CUDA BF16 training commands and encoder transfer, see
[PRETRAINING.md](PRETRAINING.md). The example below is the earlier plain masked
PatchTST baseline; it is not DropPatch or ArcTan.

Run the example from this directory:

```bash
python model_example.py --source both --steps 5
```

Or use the components in your own script:

```python
import torch
from torch.utils.data import DataLoader
from dataset import RoadDataset
from patchtst import PatchTST, PatchTSTPretrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
train = RoadDataset(source="both", split="train", window_size=1024, stride=512)
loader = DataLoader(train, batch_size=16, shuffle=True)
encoder = PatchTST(train_stats=train.train_stats)
model = PatchTSTPretrainer(encoder, loss="l1").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

model.train()
for batch in loader:
    x = batch["x"].to(device)           # Raw values; normalization is in encoder.
    valid = batch["mask"].to(device)
    optimizer.zero_grad(set_to_none=True)
    result = model(x, valid, mask_ratio=0.4)
    if not result["loss_mask"].any():
        continue
    result["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

`loss="mse"` is also supported. Output includes:

| Key | Shape / meaning |
|---|---|
| `loss` | Scalar mean L1 or MSE over hidden, originally observed values only |
| `prediction`, `target` | `[B, C, N, P]`, in z-score units |
| `patch_mask` | `[B, C, N]`; True means intentionally hidden for reconstruction |
| `loss_mask` | `[B, C, N, P]`; hidden patch AND observed sensor value |

Random masking is independent across channels. It masks approximately the
requested proportion of available patches, leaving one visible patch per
channel when possible. Channels with fewer than two available patches receive
no random reconstruction targets. All-missing inputs produce finite outputs
and zero reconstruction loss. The availability pattern has its own small
embedding; completely missing patches are excluded as attention keys.

For repeatable validation, pass a fixed boolean `patch_mask` with shape
`[B,C,N]`. `model.eval()` disables dropout but does not freeze random masks.
Hold the masks fixed when comparing checkpoints and aggregate validation loss
by the number of valid target values rather than averaging unequal batch means.

## Fine-tuning on existing per-sample labels

Reuse the pretrained encoder and attach a new head. Example: normal plus
Kaggle's four defect types, ignoring unknown labels.

```python
from patchtst import PatchTSTPredictor
from train import focal_loss, majority_patch_targets

fine_data = RoadDataset(source="real", split="train", window_size=1024,
                        stride=512, return_labels=True)
fine_loader = DataLoader(fine_data, batch_size=16, shuffle=True)
classifier = PatchTSTPredictor(model.encoder, output_dim=5, output_level="patch").to(device)
fine_optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4)

classifier.train()
for batch in fine_loader:
    x, valid = batch["x"].to(device), batch["mask"].to(device)
    target, keep = majority_patch_targets(batch, "kaggle_type", classifier.encoder.patch_length)
    target, keep = target.to(device), keep.to(device)  # [B, patches]
    if not keep.any():
        continue
    fine_optimizer.zero_grad(set_to_none=True)
    logits = classifier(x, valid)  # [B, patches, 5]
    loss = focal_loss(logits, target, gamma=2).sum() / keep.sum()
    loss.backward()
    fine_optimizer.step()
```

The patch head assigns one class to each patch. Majority voting happens in
the training code, using the original per-sample labels and input masks.
Unknown or tied patches are excluded from loss. The model can still produce
numerical predictions for those patches. The optional `output_level="sample"`
head retains per-timestamp outputs for older workflows.
This minimal fine-tuning example uses unweighted focal loss. The supervised
trainer additionally supplies `alpha` fitted from TRAIN class counts; see
[README.md](README.md#supervised-training-first) for the weighting convention.

| Task | Head configuration | Dataset target |
|---|---|---|
| Binary defect | `output_dim=2` | `defect`, or explicitly chosen `defect_assumed_normal` |
| Normal + four Kaggle defect types | `output_dim=5` | `kaggle_type` |
| Synthetic type, including pothole | `output_dim=3` | `synthetic_type` |
| Background-quality grade classification | `output_dim=4` | `quality_grade` |
| Background IRI regression | `output_dim=1` | `iri`, masked with `iri_valid` |

Grade classification here is ordinary four-class classification; an ordinal
loss/head is not implemented. An IRI head emits an unconstrained scalar; your
regression script can choose an appropriate target transform or output link.
Kaggle depression is not automatically treated as a pothole. See README.md for
the different label definitions and available supervision.

For one prediction per full window, set `output_level="window"`; output becomes
`[B, output_dim]` after masked temporal pooling of each channel's embeddings.
Supply your own defensible window/event labels; the dataset has per-sample labels.

## Normalization and checkpoint reuse

Pass `train.train_stats` once when constructing the encoder. It stores the
seven means/stds as buffers, which move to GPU and are saved with its weights.
The same saved normalizer is used when evaluating real-only, synthetic-only,
validation or test batches. No normalization is fitted during `forward()`.
Do not manually z-score inputs a second time. Without `train_stats`, the model
uses mean=0/std=1 and expects whatever scaling your caller has chosen.

```python
# Save the transferable backbone, including its fixed training normalizer.
torch.save({"config": model.encoder.config,
            "encoder": model.encoder.state_dict()}, "encoder.pt")

saved = torch.load("encoder.pt", map_location="cpu", weights_only=True)
encoder = PatchTST(**saved["config"])
encoder.load_state_dict(saved["encoder"])
classifier = PatchTSTPredictor(encoder, output_dim=5).to(device)
```

To resume an exact pretraining/fine-tuning run, also save the task head,
optimizer state and random-number-generator states in your training script.
Use the same head configuration when restoring its state dictionary.

## Temporal scope and checks

This encoder is **bidirectional**. It is for masked reconstruction and offline
classification/regression over the supplied window. It does not implement
causal next-patch forecasting. Do not use the shifted input/target example in
`example.py` with this encoder: later input patches would reveal targets.

Hidden patch values are removed before projection and are excluded from any
per-window statistics (there are no such statistics). The isolation guarantee
applies to the prepared tensors. This model does not change the dataset's
documented offline Kaggle IMU interpolation, which can use the next raw sensor
observation within a split. No new claim of strict streaming causality is made.

Tests cover patch ordering, shared channel independence, hidden-value
perturbations and gradients, missing-data behavior, supervised backward passes,
checkpoint normalization, optimization on a learnable waveform and CUDA mixed
precision when available. The example uses only TRAIN batches. These checks
establish that training works; they do not establish F1 or real-world transfer.

```bash
python -m unittest discover -s tests -v
```
