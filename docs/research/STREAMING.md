> Historical research note from the source workspace. Paths, commands, and run results may refer to earlier experiments. See [the current training guide](../TRAINING.md) and [migration notes](../MIGRATION.md) before running a command.

# Four-model ensemble and causal distillation

Experiment artifacts: `reports/ensemble_causal_20260917/`.
The models retain patch-level binary localized disturbance and IRI outputs.

```bash
# Already completed/running studies have frozen plans: do not reinitialize.
.venv/bin/python -m road_training.experiments.ensemble_teachers --init
.venv/bin/python -m road_training.experiments.ensemble_teachers --run
.venv/bin/python -m road_training.experiments.streaming_study --init
.venv/bin/python -m road_training.experiments.streaming_study --run
.venv/bin/python -m road_training.experiments.streaming_deployment
```

Use `streaming_evaluation.load_teachers` to load the ensemble receipt. It checks
all four hashes and averages disturbance probabilities and IRI predictions.
Do not average logits or independently fit validation/test normalization.

The causal model is in `streaming_model.py`. Raw SI input and boolean masks have
shape `[B,16,7]` for one new patch. Missing channels stay masked. Normalization
uses only this stream's current/past observations; no global statistics.

For a complete live interface, `live_inference.RoadStream` buffers incomplete
patches, keeps delayed target validity, and returns explicit target/emission
sample indices. It supports either the student or the rolling ensemble:

```python
from road_training.live_inference import RoadStream
from road_training.streaming_evaluation import load_teachers

ensemble = load_teachers("reports/ensemble_causal_20260917/teacher_checkpoints.json")
stream = RoadStream(ensemble, delay_patches=2)
results = stream.push(new_samples, observed_mask)  # [new_sample_count,7]
# Only use rows with valid=True. Reset stream at every new drive.
```

```python
import torch
from road_training.streaming_evaluation import load_student

model = load_student("path/to/selected/roughness/best.pt", device="cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = model.initial_state(batch_size=1)  # reset at every new drive
    for j, (x, mask) in enumerate(patch_source):
        output, state = model.stream(x.cuda(), mask.cuda(), state)
        target_patch = j - model.delay_patches
        if target_patch < 0:
            continue
        # Low-level patch_valid is emission validity; retain target validity
        # separately when delaying outputs. RoadStream handles that bookkeeping.
        probability = output["disturbance_logit"].float().sigmoid()
        iri = output["roughness"]
        # This result describes target_patch and becomes available NOW.
```

Keep inference under `no_grad`/`inference_mode`; otherwise any stateful neural
network can retain its autograd graph. State is per drive and per batch row;
do not share it across cars or reorder rows without reordering their states.
Discard incomplete 16-sample patches or buffer them until complete. Do not
invent a flush of future sensor values at drive end. Initial history is masked
padding, so the model has a cold-start period rather than real preceding data.

Training uses `ContextWindows` with 48 past patches, 64 target patches and two
future patches. A causal student only accesses the portion available by its
declared emission time. Predictions and labels are aligned explicitly by
`aligned_output`; the teacher sees the original target window. Shared sensor
augmentation is applied before cropping the teacher input.

Both supervised and distilled students use the same hard labels, focal weights,
batch composition, sampling seeds and augmentation. The distilled model adds
soft teacher probabilities and IRI targets. A separate final IRI-head stage
keeps the encoder and detector frozen, exactly as for the teachers.

Latency reports distinguish batch-size-one compute, sensor acquisition, and
the additional output delay. CPU/GPU benchmarks on this workstation do not
establish phone/embedded latency. See the experiment's `literature.md` and final
report for comparisons, evidence and limits.
