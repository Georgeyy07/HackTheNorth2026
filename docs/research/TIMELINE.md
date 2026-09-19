> Historical research note from the source workspace. Paths, commands, and run results may refer to earlier experiments. See [the current training guide](../TRAINING.md) and [migration notes](../MIGRATION.md) before running a command.

# One road timeline per drive

`RoadTimelineStream` extends rolling inference with a fixed commitment time,
same-patch context consensus, binary event hysteresis, and independent IRI
estimates. Existing frozen training and inference modules are unchanged.

```python
import json
from pathlib import Path
from road_training.streaming_evaluation import load_teachers
from road_training.timeline_stream import RoadTimelineStream

root = Path("reports")
ensemble = load_teachers(root / "ensemble_causal_20260917/teacher_checkpoints.json")
config = json.loads((root / "timeline_20260918/selection.json").read_text())["config"]
stream = RoadTimelineStream(ensemble, config, session_id="drive_001")

# new_samples and observed_mask: [new_sample_count,7], raw SI, 100 Hz.
for new_samples, observed_mask in sensor_chunks:
    for row in stream.push(new_samples, observed_mask):
        save_final_row(row)  # each absolute patch appears once
        if row["event_transition"] == "start":
            notify_event(row)

end_status = stream.finish()  # unfinished tail/event stays explicitly censored
stream.reset(session_id="drive_002")
```

`save_final_row`, `notify_event` and `sensor_chunks` above are application hooks,
not repository functions. The code does not send alerts to external services.

Each final row includes:

- `session_id`, `target_patch`, `start_s`, `end_s`: the target location on the
  drive timeline, with 160 ms resolution.
- `available_s`: when the necessary sensor data arrived, excluding compute.
- `status="final"`, `valid`: finality and observation availability are separate.
- `probability`, `disturbance`: the combined score and hysteresis decision.
- `iri_m_per_km`, `quality_grade`: estimated roughness and existing project bins
  (0/1/2/3 at thresholds 2/4/6 m/km).
- `votes`, `context_spread`: number of context estimates and their score range;
  the range is not a calibrated confidence interval.
- `event_transition`, `event_id`, and applicable event sample indices. Event
  boundaries describe the estimated target interval; alert time describes when
  it became available. They must not be substituted for each other.

`stream.provisional()` returns up to two unfinished patches for a UI that wants
to display revisions. Keep them visually distinct from committed results.
Missing sensor input produces `None` task outputs, not normal-road labels.
At drive end, the newest two patches cannot receive their full future context;
`finish()` reports them as provisional without synthesizing data.

The selected configuration averages same-patch scores at context ages 0/160/320
ms with weights 1/2/3, enters an event at 0.60 and leaves below 0.50. It imposes
no minimum event duration, so short disturbances are not automatically erased.
The IRI head uses its latest estimate without extra smoothing: validation did
not support averaging or EWMA as a default for that head.

Input chunks may have arbitrary lengths; the stream buffers fewer than 16
unconsumed samples. Preserve the 100 Hz time grid, including gaps represented
as masked samples. Do not concatenate independent drives or silently omit
missing elapsed time. State is per drive; use independent objects for cars.

The model uses 1,024 observed/padded samples, while the postprocessor retains
only two unresolved target patches and event state. Append final rows to
external storage rather than accumulating an entire drive inside the predictor.

A final prediction is an estimate, not an exact measurement. The current head
cannot recover within-patch boundaries; LiRA IRI supervision is section-level.
See [the evaluation](../reports/timeline_20260918/report.md) for precision/recall
tradeoffs, latency, evidence and limitations.
