> Historical research note from the source workspace. Paths, commands, and run results may refer to earlier experiments. See [the current training guide](../TRAINING.md) and [migration notes](../MIGRATION.md) before running a command.

# Time-RCD road pretraining

Time-RCD trains a contextual anomaly classifier with masked reconstruction as
an auxiliary task. It is not an unlabeled contrastive loss or a reconstruction
error detector. The reference is [paper v5, Section 2.2 and Appendix C.4](https://arxiv.org/html/2509.21190v5)
and the [official code at commit 372bb98](https://github.com/thu-sail-lab/Time-RCD/tree/372bb980426b2f67007311c6f3165ab789c79bef).
The upstream Apache-2.0 license is preserved in `../vendor/time_rcd/LICENSE`.

The arXiv metadata currently includes a withdrawal note, while the repository
describes ICML 2026 acceptance. We assess the released method independently of
those publication claims. The repository publishes model, masking and loss
helpers, but not the complete optimizer training driver; our loop is explicit.

## Method

1. Normalize raw `[B,T,C]` signals with fixed, saved TRAIN statistics.
2. Patch each channel into `[B,C,N,P]`, with `P=16` in this experiment.
3. Flatten channel-patch pairs into one sequence of `C*N` tokens. Attention
   connects different times **and different sensors**. With `T=1024` and seven
   channels, the complete context has 448 tokens.
4. Apply the released architecture's full-width RoPE, learned same/different
   channel attention bias, RMSNorm and gated GELU feed-forward blocks.
5. Expand each contextual patch into `P` timestep embeddings. A shared MLP
   predicts two anomaly logits per channel/timestep; average logits over
   observed channels. A separate three-layer MLP reconstructs hidden values.
6. Hide 15% of whole time patches, jointly across channels, using Gaussian
   replacement noise with standard deviation 0.1 in normalized units.

The pretraining objective is:

```text
loss = cross_entropy(anomaly_logits[known], labels[known])
     + mean_squared_error(reconstruction[hidden & observed],
                          original_values[hidden & observed])
```

Both weights are 1. Both heads use the same corrupted forward pass. Targets
are the original sensor values including genuine disturbances, not hypothetical
normal-road counterfactuals. Reconstruction stays patch-based: no isolated
timestamp masking or latent-feature targets are substituted.

The pretraining anomaly labels are timestep labels `[B,T]`; `-100` means
unknown. This is separate from our downstream majority patch labels `[B,N]`.
At clean inference there is no corruption, and the anomaly head supplies scores.
Reconstruction errors are never used as disturbance probabilities.

## Road adaptations and implementation details

Our compact encoder has width 128, three layers, four heads and FFN width 512,
versus the paper's width 512/eight-layer model. The pretraining projection
width is 64. The encoder has 794,904 parameters; pretraining has 1,011,835.
The downstream two-head road model has 1,028,378 parameters.

We use our corrected physics simulations and disturbance response labels,
rather than the authors' 2.5-billion-point generic anomaly corpus. These road
examples do not establish that we reproduce the same context-dependent
normality diversity. Fixed combined TRAIN normalization preserves physical
amplitude and matches the previous controls; upstream uses batch normalization
of values in its collation code. Our normalization also prevents hidden targets
affecting visible values through per-window statistics.

The normalizer and simulator calibration use real TRAIN information. Thus
pretraining uses synthetic gradients, but is not a strict zero-shot study.
No real labels are loaded during pretraining. Our synthetic split contains
96 training recordings and 32 validation recordings, with disjoint parent
recordings. These include multiple sensor views of physical simulations, so
the recording count is not a count of independent road geometries.

Missing gyroscopes and unknown labels stay missing. The upstream loss converts
labels to binary before checking its padding sentinel; our loss checks known
labels first. Missing channels are excluded from the channel-logit average,
attention keys and reconstruction loss. A zero-initialized availability
embedding supports partially observed patches.

For efficiency, missing tokens are packed out, preserving the original
channel-major RoPE positions. Same/different-channel attention bias is
implemented exactly by extra Q/K coordinates: its row-constant component
cancels in softmax. This permits fused attention without allocating a dense
learned attention-bias matrix. Tests compare outputs and gradients to explicit
biased attention; no approximate attention is introduced.

## Train and transfer

```bash
.venv/bin/python -m road_training.pretrain_rcd \
  --output road_training/checkpoints/rcd \
  --epochs 12 --steps-per-epoch 256 --batch-size 256
```

Defaults require CUDA BF16. AdamW uses LR `5e-4`, weight decay `1e-5`, a constant
LR and gradient clipping at 1. One epoch is a declared number of random batches
from stride-one TRAIN windows, not one exhaustive pass over overlapping data.
The experiment caps training at 12 epochs; synthetic validation selects the
lowest CE + masked MSE checkpoint, with seven-check patience. Validation masks
and noise are fixed across epochs. Clean synthetic-validation precision,
recall and F1 are also recorded but do not select the checkpoint.

```python
from road_training.rcd import load_rcd_encoder
from road_training.patchtst import PatchTSTRoadModel

encoder = load_rcd_encoder("road_training/checkpoints/rcd/best.pt")
model = PatchTSTRoadModel(encoder).cuda()
# model(x, observed) returns patch roughness and disturbance logits.
```

Encoder transfer retains attention, patch embedding and TRAIN normalization.
The pretraining timestep projection and anomaly/reconstruction heads are
discarded; our two patch-level road heads start fresh. This transfer procedure
is a downstream adaptation; the paper primarily studies direct anomaly scoring.
The checkpoint retains the entire pretrainer for that use as well.

`rcd_study/` runs three seeds of pretraining and four downstream arms: identical
RCD architecture from scratch, full fine-tuning, frozen linear probing and
encoder LR `5e-6` with head LR `1e-4`. All use the same real TRAIN inputs,
normalizers, downstream labels, losses and stopping rule. The original PatchTST
results are included as an architecture reference. Pretraining compute is not
matched to scratch training.

Results and the frozen plan live in `../reports/rcd_pretraining_20260917/`.
The runner is detached from the conversational process. It evaluates TEST only
after all validation-selected checkpoints are frozen, with threshold 0.5.
Historical TEST exposure, sparse severe-roughness labels, and the distinction
between binary disturbance and pothole-specific detection remain limitations.

## Verify

```bash
.venv/bin/python -m pytest -q road_training/tests/test_rcd.py
```

Tests cover exact attention-bias outputs/gradients, temporal and cross-channel
context, missing values and entire missing sequences, hidden-target isolation,
CE/MSE definitions, unknown labels, whole-patch masking and checkpoint transfer.
