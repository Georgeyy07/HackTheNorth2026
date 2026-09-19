# Run inference on a recorded drive

The upload flow uses the four shipped **ordinal + disturbance** PatchTST weights in
`models/ordinal_pvs/ensemble.json`. It predicts good / medium / bad road quality
and a combined localized disturbance category, with the existing fixed Kalman
filter. It does not estimate numeric IRI or distinguish potholes from cracks.

## Open the web demo

From the repository root, install the dependencies and start the server:

```bash
python -m pip install -e '.[viewer]'
npm ci --prefix road_viewer
python -m road_viewer.server --uploads artifacts/user_drives --port 8766
```

Open http://localhost:8766, select a CSV under **Replay your own drive**, select
its units, and press **Run inference**. CUDA with BF16 is used when available;
otherwise inference runs on CPU. Use `--device cuda` to require CUDA explicitly.
The upload is processed on the server before replay starts. Playback then shows
predictions at their recorded availability times; this is a recorded-drive demo,
not a live phone socket. Hardware/network latency is not simulated.

To retain the original demo drives, also pass `--export artifacts/test_drive_inference`.
Imports persist across server restarts in the uploads directory. One inference
job runs at a time, with a limit of 64 MiB and 60 minutes per CSV.

## CSV contract

One row per sensor timestamp, strictly increasing with no duplicates. Example:

```csv
time_s,accel_x,accel_y,accel_z,speed,latitude_deg,longitude_deg
0.00,0.12,-0.03,9.81,8.5,39.639,22.419
0.01,0.15,-0.02,9.86,,,
0.02,0.11,-0.04,9.79,,,
```

This illustrates the schema; provide at least 16 samples at 100 Hz for inference.

| Field | Required | Default units / meaning |
| --- | --- | --- |
| `time_s` | Yes | Relative seconds or Unix seconds; selectable milliseconds or ISO timestamps |
| `accel_x`, `accel_y`, `accel_z` | Yes | m/s² **including gravity**; selectable g |
| `speed` | No | m/s; selectable km/h |
| `latitude_deg`, `longitude_deg` | No | GPS degrees; supply both only when a new fix arrives |
| `gyro_x`, `gyro_y`, `gyro_z` | No | rad/s, displayed only; never consumed by this model |

Acceleration must already use the nominal vehicle frame **x forward, y left,
z up**. A stationary level sensor should have approximately +9.81 m/s² on z.
Convert phone axes using your mounting calibration before exporting. This importer
does not infer arbitrary phone orientation or add gravity to linear acceleration.
See [the data contract](DATA.md) for training frame limitations.

Common aliases `timestamp`, `time`, `ax`, `ay`, `az`, `speed_mps`, `lat`, `lon`,
`latitude`, and `longitude` are accepted. For other headers, use the web form's
column mapping or a JSON file with canonical names as keys:

```json
{"time_s":"timestamp_ms","accel_x":"phone_forward","accel_y":"phone_left","accel_z":"phone_up"}
```

Missing values are empty cells, not zero placeholders. Missing speed is masked;
missing GPS leaves the map empty while sensor plots and inference continue.
The mask permits inference without speed, but accuracy on such inputs has not
been established by this integration test. Repeated old GPS coordinates marked
as new rows would incorrectly refresh their age; leave those cells empty instead.

## Timing and post-processing

- Inputs are resampled to 100 Hz using **past-only sample holds**. IMU readings
  expire after 50 ms; speed and GPS expire after 3 s. GPS is never interpolated.
  Prefer recordings near 100 Hz: resampling cannot recover missing bandwidth.
- Each rolling input has up to 10.24 s of context. The beginning is masked on
  the left; early predictions have less context and may be less reliable.
  The model applies instance normalization plus its learned statistics branch.
- Patches span 160 ms. Each patch receives up to three context predictions,
  weighted 1:2:3. Finalization waits for two subsequent patches (320 ms after
  the target ends). Bidirectional attention sees only the input available at
  the current processing step.
- Ordinal class probabilities are averaged across models and contexts. The
  decision uses the ordinal median, matching the model's evaluation protocol.
  Final disturbance scores use Kalman Q/R = 3.2, onset 0.70, offset 0.50.
  These filtered scores are not calibrated probabilities. Kalman is applied
  to disturbance scores, not road-quality class probabilities.
- A patch with missing accelerometer XYZ samples is marked invalid. GPS for
  a prediction uses the last fix known at the **target patch start**.
- Every resampled sensor timestep is exported. The last two complete patches
  remain provisional; an incomplete final patch is retained as sensor data
  without a prediction. No extra future samples are invented to finish a drive.

The ordinal reference boundaries for LiRA are approximately 1.4994 and 2.6831
m/km, with PVS's original good / regular / bad labels also used in training.
They differ from the old regression demo's 2/4/6 m/km display bins. The ordinal
viewer has three classes and does not display a fourth “terrible” category.

## Command-line inference and output files

```bash
python -m road_training.infer_csv drive.csv --output artifacts/my_drive \
  --time-unit ms --acceleration-unit m/s2 --speed-unit km/h \
  --columns mapping.json --device auto
python -m road_viewer.server --export artifacts/my_drive --port 8766
```

The output directory must not already exist. Omit `--columns` for standard
headers. A successful export contains:

- `manifest.json`: session metadata, input units, source and ensemble hashes.
- `<session>/samples.parquet`: every 100 Hz sensor sample, missing values retained.
- `<session>/gps_fixes.parquet`: observed GPS coordinates and relative timestamps.
- `<session>/updates.jsonl.gz`: provisional/final predictions with target time,
  availability time, class probabilities, filtered/raw disturbance scores,
  event transitions, and target GPS. The web **Inference data** button downloads
  this file. Sensor data remain in the separate Parquet files.

Numeric Unix and ISO inputs retain their absolute origin in the manifest;
relative timestamps display as elapsed drive time. For programmatic streaming,
`road_training.ordinal_stream.OrdinalRoadStream.push(x, mask)` accepts chunks
of canonical 100 Hz `[samples, 4]` input and returns the same timed updates;
call `reset()` between drives. CSV parsing and GPS association live in
`road_training.infer_csv`.
