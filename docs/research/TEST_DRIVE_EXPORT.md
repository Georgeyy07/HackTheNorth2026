> Historical research note from the source workspace. Paths, commands, and run results may refer to earlier experiments. See [the current training guide](../TRAINING.md) and [migration notes](../MIGRATION.md) before running a command.

# Complete test-drive inference for map replay

The export is at `reports/test_drive_inference_20260919/`. Its `manifest.json`
lists every drive and the coverage audits. It contains all nine real TEST
recordings: Kaggle's held-out Larisa interval and eight LiRA M13 passes.
RoadSens has no held-out recordings in this project's split and is not part of
this export. No supervised-window filter or annotation-boundary filter is used.

Reproduce into a **new** output directory:

```bash
.venv/bin/python -m road_training.export_test_drives \
    --output reports/test_drive_inference_another_run
```

The exporter verifies source files, the prepared data, and the four checkpoints
against their recorded SHA-256 hashes. It uses the seed 52–55 mounting-augmented
ensemble in `reports/mounting_ensemble_20260919/ensemble.json`, CUDA bfloat16,
and batch size 1. Each drive starts with its own empty stream. Model training,
normalization, thresholds, and ensemble weights are unchanged.

## Files in each drive directory

| File | Use |
| --- | --- |
| `samples.parquet` / `samples.csv.gz` | Every 100 Hz model-input sample, clocks, raw SI sensor values, masks, GPS, and its patch's latest exported prediction. |
| `patches.parquet` / `patches.csv.gz` | One prediction per absolute 16-sample patch, including the incomplete final patch if present. Includes target location, emission location, availability time, and event transitions. |
| `updates.jsonl.gz` | Chronological replay messages: provisional revisions and each final patch emitted exactly once. A UI applies updates by `(session_id, target_patch)`. |
| `events.json` | Committed event intervals, alert times, GPS near the first target patch, peak scores, and censored endings. |
| `gps_fixes.parquet` / `gps_fixes.csv.gz` | Original observed GPS fixes with their timestamps and coordinates. |
| `imu_native.parquet` | Original irregular sensor observations, in SI units, before resampling. Kaggle has six IMU channels; LiRA has three accelerometer channels. |
| `speed_native.parquet` | Original observed speed and timestamps, in m/s. |
| `route.geojson` / `events.geojson` | Whole-drive route and located events for a static preview. These contain future information relative to an in-progress replay; use their timing sidecars or `updates.jsonl.gz` to reveal results live. |
| `ground_truth.parquet` / `reference_annotations.json` | Separate original labels for comparison. Never consumed by inference or GPS processing. |
| `session.json` | Recording provenance, clock origin, sensor availability, end-of-drive status, and audit. |

Parquet and CSV represent the same tables. CSV nulls are empty fields; nullable
booleans are `True`, `False`, or empty. JSON uses standard `null` and contains no
NaN/Infinity. The JSONL stream is gzip compressed UTF-8 with one object per line.

## Replay clocks and coordinates

All `*_s` replay fields are seconds from the first prepared sample of this TEST
recording, unless explicitly prefixed `source_`. All `*_unix_ms` fields are Unix
milliseconds, including fractional milliseconds; they can be used by JavaScript
without a nanosecond-sized integer. The exact origin in `session.json` is also
retained as a decimal-string Unix nanosecond value. Calendar times come from
source timestamps, rather than being guessed from recording filenames.

For a moving-car demo, use `gps_fixes.time_s` or the causally aligned GPS on
`samples.time_s`. Use native sensor timestamps to show original measurements;
if showing the model's resampled sensors, wait until `input_available_s`.

Reveal each inference update only when `available_s <= replay_clock_s`.
Its `start_s` and `end_s` identify the road interval being classified, with an
exclusive end. The car will already be farther along when that result appears.
Use `target_latitude_deg` / `target_longitude_deg` to mark the road interval;
`emission_latitude_deg` / `emission_longitude_deg` describe the later car location.
The target coordinate is the most recent GPS fix at the target patch's midpoint
(`gps_target_time_s`). An event marker uses the midpoint of its first patch.

GPS is held from the most recent **already observed** fix, with no future-value
interpolation. `gps_fix_index` points into `gps_fixes`; `gps_age_s` reports age.
Coordinates are null before the first fix or when a fix is over three seconds
old. GPS dropouts do not remove sensor samples or stop inference. The raw fix
rate and age limit localization accuracy; held GPS coordinates are not exact
per-sample road positions. LiRA uses the measured `gps` field, not the offline
`gps_mapmatch` field or the pavement survey's coordinates. GeoJSON uses
`[longitude, latitude]`, while table column names spell out the axes.

```python
import gzip
import json
from pathlib import Path
import pandas as pd

root = Path("reports/test_drive_inference_20260919")
catalog = json.loads((root / "manifest.json").read_text())
drive = root / catalog["sessions"][0]["session_id"]
samples = pd.read_parquet(drive / "samples.parquet")

# Feed these objects into a replay queue ordered by available_s.
with gzip.open(drive / "updates.jsonl.gz", "rt") as f:
    updates = [json.loads(line) for line in f]

replay_clock_s = 10.
visible_updates = [u for u in updates if u["available_s"] <= replay_clock_s]
latest_by_patch = {u["target_patch"]: u for u in visible_updates}
committed = [u for u in latest_by_patch.values() if u["is_final"]]
```

The availability clock includes sensor arrival and the post-processing delay;
it excludes GPU execution, network transport, and rendering. Add those costs to
simulate a particular device. A nominal sample interval ends at `(index+1)/100`.
Kaggle's existing 100 Hz preparation interpolates nearby native IMU observations
within TEST. The exporter explicitly waits for the interpolation's right-hand
observation if needed. LiRA preparation holds past observations causally.

## Prediction meanings and the unfinalized tail

The context is 1,024 samples / 10.24 seconds, ending at the current rolling patch.
The stride is 16 samples / 160 ms. The model's masked instance normalization
remains inside the model; exported accelerations include gravity in m/s²,
gyroscope readings are rad/s, and speed is m/s. Stored recording axes are
unchanged. LiRA does not contain gyroscope measurements: those sample columns
are null with false masks. Filling nulls with zero and applying their masks
reproduces the exact prepared model inputs.

The existing validation-selected Timeline config is reused unchanged:

- Three estimates of the same target patch at successive context ages are
  combined with weights 1:2:3 for the disturbance probability.
- A disturbance starts at probability 0.60 and ends below 0.50.
- Roughness uses the latest estimate, without extra temporal smoothing.
- A final patch is committed after two following patches, nominally 320 ms
  after its end, or 480 ms after its start. It is not subsequently revised.

This config was selected for the preceding ensemble, then applied to the newly
trained ensemble without tuning on TEST. Startup context is masked until enough
real observations have arrived; startup estimates can be less reliable.

`probability` is the localized-disturbance score. `disturbance` is its final
hysteresis decision, not a five-class prediction. It includes manholes, cracks,
bumps, and depressions and does **not** specifically identify a pothole.
`event_id` on the sample table is populated only inside an active final event.
Patch/update rows can carry the preceding event ID on an `end` transition; that
closing patch is already classified as non-disturbance.

`iri_m_per_km` is estimated overall roughness. `quality_grade` is 0/1/2/3 at
thresholds 2/4/6 m/km, with project display names good/medium/bad/terrible. These
are project display bins, not a universally validated road-rating scale.
LiRA supplies section-level IRI supervision; 160 ms outputs do not imply
equally precise ground-truth roughness resolution. The roughness head on Kaggle
and the disturbance head on LiRA have no corresponding ground truth in those
datasets, so the exported outputs cannot establish their accuracy there.

Every sample has a model estimate, but the last two complete patches remain
`status="provisional"`. If the recording ends inside a patch, that partial patch
gets one EOF estimate using its observed samples and an explicitly masked
unobserved remainder, with `status="provisional_partial"`. It does not update
Timeline or finalize earlier patches. This EOF fallback is not separately
validated. Provisional rows expose scores and roughness but leave the binary
decision and event ID null. No future sensor samples are invented.

`is_final` distinguishes commitment from `prediction_valid` / patch `valid`,
which indicate observed inputs. Entirely missing patches yield null estimates,
not normal-road classifications. An event still active at the last final patch
has `end_s=null`, `end_censored=true`, and a `classified_until_s` boundary;
its true ending must not be inferred from the end of the recording.

Each patch estimate is repeated on its constituent sample rows for convenient
joining. This gives complete sample coverage, not 100 independently inferred
predictions per second. `context_spread` measures disagreement between rolling
contexts; it is not calibrated uncertainty.
