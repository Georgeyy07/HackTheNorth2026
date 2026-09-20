> Historical research note from the source workspace. Paths, commands, and run results may refer to earlier experiments. See [the current training guide](../TRAINING.md) and [migration notes](../MIGRATION.md) before running a command.

# Instance-normalized road training

`instance_model.py` supplies a supervised bidirectional model that normalizes
each input window separately. It does not use training-set means, standard
deviations or running normalization buffers.

```python
from road_training.instance_model import InstancePatchTST, InstanceRoadModel

model = InstanceRoadModel(
    InstancePatchTST(max_patches=64),
    statistics_mode="both",  # "none", "roughness", or "both"
)
output = model(x, mask)  # x/mask: [B,1024,7]; outputs: [B,64]
```

For each window and channel, observed time samples determine a mean and
population variance. Normalization uses `(x - mean) / sqrt(variance + 1e-5)`
in FP32, then missing entries become zero. Missing values never enter those
statistics. Completely missing channels stay zero. Statistics from other
windows, recordings, batches or splits are never used.

Full-window statistics are appropriate for this bidirectional supervised
task. They include the observed context on both sides of each patch within
the input. Hidden-patch reconstruction is explicitly rejected because it
would require computing statistics only from visible input values.

Pure instance standardization removes absolute amplitude and speed level.
The optional statistics branch preserves three descriptors for each channel:
`asinh(window_mean)`, `log(window_standard_deviation)`, and observed fraction.
A small shared embedding adds these to the roughness head's features, or to
both heads' features. The encoder still receives instance-normalized patches.
These descriptors come from the same window and involve no dataset fitting.

`instance_loss.py` computes Kaggle and RoadSens focal losses separately using
their own TRAIN class counts and known-label masks. Each dataset has its own
mean before combining tasks. Unknown RoadSens labels never become normal
targets. LiRA supplies IRI Huber loss; no road-type target is invented for it.

The study runner, `instance_study.py`, uses all three real datasets and the
existing rotation/noise augmentation. It screens five declared recipes with
seed 42, confirms the best two using seeds 43 and 44, then freezes the winning
three checkpoints before test evaluation. Selection uses Kaggle validation
binary disturbance F1 at threshold 0.5; IRI error remains separately visible.

```bash
.venv/bin/python -m road_training.experiments.instance_study --init
.venv/bin/python -m road_training.experiments.instance_study --run
```

The initializer deliberately refuses to overwrite a frozen study. The runner
resumes interrupted runs. Per-epoch histories, TensorBoard events, checkpoints
and provenance are under `reports/instance_roadsens_20260917/`.

The completed study's recommended two-head checkpoint is
`reports/instance_roadsens_20260917/best.pt`. Its seed was selected by validation
F1. The final roughness head was adapted on LiRA TRAIN with the encoder,
window-statistics branch and disturbance head frozen. That step changed no
disturbance predictions. Its separate CLI is `instance_roughness.py`.

```python
import torch

saved = torch.load("reports/instance_roadsens_20260917/best.pt",
                   map_location="cpu", weights_only=False)
config = saved["config"]
model = InstanceRoadModel(InstancePatchTST(**config["encoder_config"]),
                          **config["model_config"])
model.load_state_dict(saved["model_state"])
model = model.cuda().eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(x.cuda(), mask.cuda())  # raw SI values, no external z-score
probability = output["disturbance_logit"].sigmoid()
iri = output["roughness"]
```

The dataset supplies seven channels: acceleration XYZ, gyro XYZ, and available
speed. Missing channels remain zero with a false mask. The selected model
accepts 1,024 time samples and returns 64 roughness values and disturbance
logits per window. See the [final results](../reports/instance_roadsens_20260917/final_report.md).

This changes the training recipe, not the dataset splits. Kaggle keeps its
purged chronological Larisa segments, LiRA keeps M3/M13 as validation/test,
and RoadSens is TRAIN-only. The benchmark does not establish performance on
an unseen RoadSens drive or city/device. Earlier checkpoints remain intact.

Method context: [PatchTST](https://arxiv.org/abs/2211.14730) and
[RevIN official implementation](https://github.com/ts-kim/RevIN).
Our supervised task has no inverse normalization of IRI or class logits;
this is an input-normalization adaptation, not a forecasting reproduction.

The new [phone-mounting recipe](MOUNTING.md) uses the same model with aligned
RoadSens data and wider, configurable yaw/tilt augmentation. Its runner is
`mounting_study.py`; the completed study above and its checkpoints are unchanged.
