# Fleet IMU and model output contract

Imported and verified: **851,938 sensor samples**, **53,207 finalized predictions**, and **46,180 map rows with matching inference and sensor data** across the two scenarios (eight virtual car IDs). Session1 is excluded.

The active sessions2–5 scenarios use these Tiger objects in the `public` schema:

- `simulated_car_observations`: existing map records, unchanged.
- `simulated_car_imu_samples`: 100 Hz model-input acceleration and speed, including intervals without usable GPS.
- `simulated_car_inference`: finalized 160 ms patch predictions, including intervals without usable GPS.
- `simulated_car_observations_enriched`: existing map rows joined to exact-timestamp predictions and the latest IMU sample at or before the row timestamp (maximum age 20 ms). The actual `imu_sample_timestamp` is exposed; these acceleration values are individual samples, not patch averages.

All objects use the same case-sensitive `"carID"` and scenario `"timestamp"` as the map. Source session and source seconds are retained. Session1 is excluded. Virtual-car duplicates exist under distinct car IDs; do not treat them as independent real drives.

## Units and fields

`accel_x`, `accel_y`, `accel_z` are the saved model inputs in **m/s²**, VQF vehicle frame: forward, left, up. They **include gravity**; z near 9.8 at rest is expected. These are prepared/resampled model-input samples, not raw phone-axis measurements. `speed_mps` is metres/second; multiply by 3.6 for km/h. Missing speed stays SQL NULL. `settling` indicates orientation stabilization.

`p_good`, `p_medium`, `p_bad` are calibrated probabilities in that order and sum to one. The saved calibration bias was applied once; do not calibrate again in the UI. `defect_probability_kalman` is the saved post-consensus Kalman-filtered probability (0–1), not the Boolean flag. `defect_probability_raw` retains the pre-filter consensus score. Model checkpoint hash and calibration ID are recorded on every prediction.

`imu_defect_detected` is the saved hysteresis state (onset .7, offset .5), so it is **not** equivalent to comparing every filtered probability to .7. Use the stored Boolean for consistency with the map. Invalid predictions retain NULL outputs and `valid=false`, never invented zeros. Only finalized predictions are imported; provisional/incomplete tails are omitted.

## Timestamp semantics

The scenario clock begins at `2026-09-20T12:00:00Z`. Use its simulated time, not the browser's wall-clock time.

```
scenario timestamp = car launch timestamp
                   + source absolute timestamp - car source start timestamp
```

For sensor rows, `timestamp` is the sample time; `available_at` preserves input availability. For inference, `timestamp` is the **target patch end**, `patch_start` is its beginning, and `available_at` is when the finalized output became available. The interval is `[patch_start, timestamp)`. The saved outputs have an availability delay of **320–343 ms**. Results are never shifted to a different road segment to hide that delay.

The existing map/video replay is retrospective: select predictions by target `timestamp` for alignment. For a causal live-style display, only expose outputs with `available_at <= map clock` and plot each at its target time. The newest result may describe an earlier road position; this is expected. If fetching via target-time windows, include enough earlier rows to cover availability delay, or query the availability column directly.

Source recording gaps are preserved. Do not draw lines across session changes, invent zero readings, or hold old scores indefinitely during a gap. Cars end at their manifest route end. Some final results become available just after that end; those remain stored with their true availability times.

## Query examples

Use bound parameters on the FastAPI backend; keep database credentials out of browser code. Filter by exact car ID and fetch small time windows (e.g. 5–10 seconds), rather than downloading the full fleet.

100 Hz graph input:

```sql
SELECT "timestamp", available_at, source_session,
       accel_x, accel_y, accel_z, speed_mps, settling
FROM public.simulated_car_imu_samples
WHERE "carID" = %(car_id)s
  AND "timestamp" >= %(from)s AND "timestamp" < %(to)s
ORDER BY "timestamp";
```

Aligned model graph:

```sql
SELECT "timestamp", patch_start, available_at, source_session,
       p_good, p_medium, p_bad, defect_probability_kalman,
       road_quality, imu_defect_detected, valid, settling
FROM public.simulated_car_inference
WHERE "carID" = %(car_id)s
  AND "timestamp" >= %(from)s AND "timestamp" < %(to)s
ORDER BY "timestamp";
```

Live-style arrivals: use `available_at >= %(from)s AND available_at < %(to)s` and `ORDER BY available_at, "timestamp"` instead. Use the same availability filter for sensor arrivals if showing acquisition delay.

One query for map observations plus aligned readouts:

```sql
SELECT * FROM public.simulated_car_observations_enriched
WHERE "carID" = %(car_id)s
  AND "timestamp" >= %(from)s AND "timestamp" < %(to)s
ORDER BY "timestamp";
```

Serve timestamps as UTC ISO8601 strings, probabilities as numbers, booleans as booleans, missing values as JSON null. Keep road quality strings lowercase. Use the same clock as [video synchronization](fleet-video-sync.md).

## Reproduction and verification

`scripts/import_fleet_telemetry.py` exports source data after verifying hashes against the active fleet manifest, then imports both tables atomically with conflict checks. The existing seven-column table is unchanged, so its importer remains compatible. Receipts and CSV exports are in `/home/origami/pothole/reports/fleet_sessions_2_5/telemetry_*`. The importer checks every existing map row against the saved probability record's timestamp, quality category and defect flag, then verifies the committed enriched view through a separate connection.
