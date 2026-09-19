> Historical research note from the source workspace. Paths, commands, and run results may refer to earlier experiments. See [the current training guide](../TRAINING.md) and [migration notes](../MIGRATION.md) before running a command.

# Patch-reconstruction pretraining

The labeled synthetic Time-RCD method is documented separately in
[RCD.md](RCD.md), including its anomaly CE + auxiliary patch-reconstruction
objective and the controlled downstream comparison.

Two methods are implemented in plain PyTorch. Neither needs road labels.
Both accept raw `RoadDataset` tensors `x, mask: [B,T,7]`, apply fixed TRAIN
z-scores, and reconstruct the sensor samples **inside non-overlapping patches**.
Missing measurements (including LiRA's absent gyroscope) never become targets.
Do not normalize the inputs again outside the model.

Files:

* [droppatch.py](droppatch.py): drop/mask sampling and masked reconstruction.
* [arctan.py](arctan.py): DiT encoder/decoder, corruption and velocity loss.
* [pretrain.py](pretrain.py): shared training loop and encoder checkpoint loading.

## DropPatch

Source: Qiu et al., [Enhancing Masked Time-Series Modeling via Dropping
Patches](https://arxiv.org/html/2412.15315v1), AAAI 2025, Method and Appendix C.
Also checked the [authors' code](https://github.com/qityy/DropPatch/blob/bb475cca0c2490d3ff59a319bd8718f624231474/models/dm.py)
at commit `bb475cca0c2490d3ff59a319bd8718f624231474`.

1. Patch each channel separately: `[B,T,C] -> [B,C,N,P]`.
2. Randomly **drop 60%** of patches, independently per channel. They do not
   enter attention and are not reconstruction targets.
3. Randomly **mask 40% of retained patches**, replacing their values with zeros
   before embedding. Keep their original temporal positional embeddings.
4. Feed all retained tokens (visible and masked) through PatchTST.
5. Use a linear `D -> P` reconstruction head. Minimize **MSE only on masked,
   originally observed values**. The target tensor is detached.

For a 1,024-sample window and patch length 16: 64 original patches become
25 retained patches, with 15 visible and 10 masked. Attention actually runs
on 25 tokens per channel. `kept_indices: [B,C,K]` records original positions;
`prediction/target/loss_mask: [B,C,K,P]` follow that retained order.

MAE here means *masked autoencoding*, not a mean-absolute-error loss. The
DropPatch paper uses MSE. It also retains masked tokens inside the encoder;
it is not the image-MAE architecture that encodes only visible tokens.

Road-specific adaptations: reuse our linear patch embedding, learned absolute
positions, existing Transformer and fixed TRAIN statistics. The authors' code
uses a convolutional embedding and instance normalization. Those are not
silently substituted into our supervised backbone. Fixed statistics preserve
physical amplitude and prevent hidden patches affecting visible values via
per-window means/variances. Missing data is handled explicitly: sampling ratios
apply to available patches; keep at least two if possible, otherwise use no
reconstruction target. Padded retained slots are excluded from attention/loss.

At fine-tuning, discard the reconstruction head and use every input patch.
No dropping or masking happens in the transferred encoder.

## ArcTan Diffusion

Source: the supplied local paper
[ArcTan Diffusion: Simplifying Time Series Pretraining by Reparameterizing the
Denoising Objective](../12636_ArcTan_Diffusion_Simplif.pdf), Sections 3.2–3.6,
Equations (1)–(4), Appendix A. This is an implementation from the manuscript,
not a reproduction of its reported benchmark scores. AdaLN-Zero is also checked
against the [original DiT implementation](https://github.com/facebookresearch/DiT/blob/main/models.py).

The network predicts clean **raw patch values in normalized units**, not latent
embedding targets. For a clean patch `x`, sample Gaussian noise `epsilon` and
`t ~ Uniform(0,1)`:

```text
z       = t*x + (1-t)*epsilon
x_hat   = decoder(encoder(z, t), t)
denom   = max(1-t, 0.05)
v       = (x     - z) / denom
v_hat   = (x_hat - z) / denom
e       = v - v_hat
loss(e) = alpha * [-a/(2*k) * log(1 + k²*e²) + a*e*atan(k*e)]
          + beta*abs(e) + gamma*v_hat²
```

Use the paper's constants `a=1.1, k=1.3, alpha=0.5, beta=0.5, gamma=0.05`.
Average over all originally observed patch components. Loss arithmetic stays
FP32 under BF16 autocast. Both target and predicted velocities use the clamped
denominator. Substituting `v=x-epsilon` near `t=1` would change the objective.
The gamma term is on **predicted velocity**, so even an exact prediction can
have nonzero total loss. Also log clean-patch reconstruction MSE separately.

The encoder and decoder use per-token timestep conditioning through
sinusoidal embeddings, an MLP, and AdaLN-Zero gates in each attention/FFN block.
The latent projection `Linear(LayerNorm(encoder_output))` belongs to the encoder
and is kept for downstream training. The decoder has separate parameters and
is discarded. At downstream inference, use the **clean endpoint `t=1`**;
there is no reverse diffusion loop, sampled noise, or decoder.

Two paper-supported tokenization modes:

| Setting | Token contents | Encoder output |
|---|---|---|
| `channel_mode="mixed"` (default, paper classification) | All `C*P` sensor values in one patch | `[B,1,N,D]` |
| `channel_mode="independent"` (paper forecasting) | `P` samples from one channel, shared encoder | `[B,C,N,D]` |

Reconstruction outputs are `[B,C,N,P]` in both modes. `timesteps` is `[B,F,N]`,
where `F=1` for mixed and `F=C` for independent. Default `timestep_mode="token"`
draws a separate level per token, as in the paper's classification setup.
`"sequence"` shares one level across each physical window and its channels.

Architecture defaults follow the paper: width 128, two encoder layers, two
decoder layers, 16 heads, FFN width 512, dropout 0.1. Our road preset uses
16-sample patches rather than the paper's default 8, plus fixed TRAIN
normalization and an availability embedding instead of instance normalization.
Absolute positions are learned. Standard DiT zero initialization is used for
residual modulation and the final output layer. No unreported diffusion schedule,
auxiliary loss, extra masking or noise-target prediction is introduced.

This encoder is different from PatchTST. A fair downstream comparison therefore
needs a **randomly initialized ArcTanEncoder control**, in addition to the
existing PatchTST baseline; otherwise architecture and pretraining effects mix.

## Training

From the repository root, using the corrected simulator corpus by default:

```bash
.venv/bin/python -m road_training.pretrain --method droppatch \
  --source both --output road_training/checkpoints/pretrain_droppatch

.venv/bin/python -m road_training.pretrain --method arctan \
  --source both --output road_training/checkpoints/pretrain_arctan
```

Choose `--source real`, `synthetic`, or `both`. `--real-dataset kaggle|lira|all`
filters real data; `--data-root` selects a different prepared corpus.
`both` samples uniformly over the chosen TRAIN windows; it does **not** enforce
the prior supervised experiment's 215/41 real/synthetic batch quota.
Unlabeled TRAIN recordings are usable. No labels are loaded at all.

Defaults: CUDA BF16, batch 256, LR `1e-4`, AdamW weight decay `.01`, gradient
clipping at 1, cosine LR decay, 50 epochs. Each epoch samples 512 batches with
replacement from stride-one TRAIN windows; it is a fixed update budget, not an
exhaustive pass over every overlapping window. These batch/update settings are
our road-training choices, not the papers' benchmark recipes. An explicit
`--device cpu` is available for small tests.

VAL uses non-overlapping windows from its existing split, at most 4,096 evenly
spaced windows by default (`--val-windows 0` uses all). It uses the encoder's
saved TRAIN normalizer, never its own fitted statistics. Corruption RNG resets
each validation epoch, giving fixed masks/noise/timesteps for the same loader
configuration without consuming TRAIN's corruption or dropout RNG. Change
`--val-source` if needed; default follows `--source`.

`config.json` records source filters, actual TRAIN/VAL recordings, validation
indices, manifest hash, normalization and architecture. `history.json` and the
terminal report per-epoch TRAIN/VAL loss and reconstruction MSE. Metrics weight
each observed target scalar equally. `best.pt` selects minimum VAL objective;
`last.pt` stores the final epoch. Both include full pretrainer and encoder-only
weights; these are weight exports, not optimizer-resume checkpoints. Existing
output directories are refused. There is no TEST option and no F1 calculation
during unlabeled pretraining. A lower pretraining loss is not evidence of a
higher downstream F1, and the two methods' objective scales differ.

Small executable check, not a performance experiment:

```bash
.venv/bin/python -m road_training.pretrain --method droppatch --source both \
  --epochs 2 --steps-per-epoch 3 --val-windows 32 --workers 0 \
  --output road_training/checkpoints/droppatch_smoke
```

## Use in a training script

```python
from road_training import RoadDataset, PatchTST, DropPatchPretrainer
from road_training.arctan import ArcTanEncoder, ArcTanPretrainer

data = RoadDataset(root="road_training/data_transfer_v2_mount", source="both",
                   split="train", window_size=1024, stride=1)
drop = DropPatchPretrainer(PatchTST(train_stats=data.train_stats)).cuda()
diffusion = ArcTanPretrainer(ArcTanEncoder(train_stats=data.train_stats)).cuda()

# In your loop: batch from an ordinary DataLoader(data).
result = drop(batch["x"].cuda(), batch["mask"].cuda())  # or diffusion(...)
result["loss"].backward()
```

Attach the existing two patch-level road heads to either saved encoder:

```python
from road_training.pretrain import load_pretrained_encoder
from road_training.patchtst import PatchTSTRoadModel

encoder = load_pretrained_encoder("road_training/checkpoints/pretrain_arctan/best.pt")
model = PatchTSTRoadModel(encoder).cuda()
# model(x, mask) -> roughness [B,N], disturbance_logit [B,N], patch_valid [B,N]
# Optimize all model parameters on labeled TRAIN data, using the existing
# patch_targets/run_epoch helpers in road_training.train_multitask.
```

The heads infer encoder feature width, including mixed-channel ArcTan latents.
Keep the saved normalizer when moving from mixed pretraining to real fine-tuning.
The existing supervised CLI still creates a fresh encoder; explicitly load the
pretrained encoder as above in your fine-tuning script. Do not assume it detects
pretrained weights automatically.

### Frozen linear probing and slower encoder adaptation

The comparison also tests two ways to protect pretrained features from
overfitting. Both use the same labeled TRAIN windows and VAL stopping rule.

```python
from road_training.pretraining_adaptation import FrozenLinearRoadModel, make_optimizer

# Strict linear probe: only two Linear layers receive gradients.
encoder = load_pretrained_encoder("path/to/selected.pt")
model = FrozenLinearRoadModel(encoder).cuda()
optimizer = make_optimizer(model, "linear_probe", head_lr=1e-4)

# Alternatively, retain the usual MLP heads and adapt the encoder slowly.
encoder = load_pretrained_encoder("path/to/selected.pt")
model = PatchTSTRoadModel(encoder).cuda()
optimizer = make_optimizer(model, "low_lr", head_lr=1e-4)
# optimizer groups: encoder=5e-6, heads=1e-4; AdamW weight decay=.01.
```

The linear probe keeps the encoder in evaluation mode even after `model.train()`;
encoder dropout is disabled, and the optimizer excludes encoder parameters.
It concatenates the channel features per patch and trains one linear IRI readout
(with a nonnegative softplus link) and one linear disturbance logit. Trainable
counts are 1,794 for the current DropPatch encoder and 258 for ArcTan. The
low-LR variant trains the full model with a fixed 20:1 head/encoder LR ratio.
Each variant keeps the saved TRAIN normalizer.

The three-seed experiment is in `pretraining_study/`; its frozen recipe and
results are under `reports/pretraining_comparison_20260916/`. DropPatch uses
epoch 12. To reproduce that shortened run while retaining its original cosine
schedule, use `--epochs 50 --stop-after-epochs 12`. Linear probing changes both
head capacity and encoder trainability, so it is not an isolated freezing test.

## Verification

```bash
.venv/bin/python -m pytest -q road_training/tests/test_pretraining.py road_training/tests/test_pretraining_adaptation.py
```

Tests check physical removal from attention, original positional indices,
hidden/dropped-target isolation including input gradients, missing channels,
analytical loss values/derivatives, near-clean denominator behavior, token and
sequence sampling, learnability, clean-endpoint encoder transfer and compatible
backpropagation through both road heads. They validate implementation, not
downstream transfer performance.
