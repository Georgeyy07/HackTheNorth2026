> Active presentation update: session1 is excluded. Use `/home/origami/pothole/reports/fleet_sessions_2_5` (46,180 rows, four cars per variant). The previous five-session Tiger rows were atomically replaced. See [video synchronization handoff](fleet-video-sync.md) for the current starts, clocks, and video URLs. Older counts below describe the previous bundle.

# Synthetic multi-car scenarios

Run the generator from the repository root:

```bash
python scripts/generate_fleet_scenarios.py \
  --imu /home/origami/pothole/reports/local_imu_sessions_20260920 \
  --vision /home/origami/pothole/reports/fleet_vision_snapshot_20260920 \
  --output /home/origami/pothole/reports/fleet_scenarios_20260920_v2 \
  --start 2026-09-20T12:00:00Z --run-id waterloo-demo \
  --cars 5 --headway 20 --stop-speed 0.3 --stop-seconds 2
```

Choose a new output directory and run ID when changing inputs/settings. The generated manifest is the audit trail, and the bundle README explains import and timing semantics.

Two independent variants are produced, each with five cars:

| Variant | Start behavior | Rows |
| --- | --- | ---: |
| staggered | Same launch timestamp, one stationary start in each of sessions 1–5 | 25,941 |
| cascade | Same route from session1, delays of 0/20/40/60/80 seconds | 39,365 |

Starts are selected automatically from observed speed. Missing speed and missing sample times break a stop. Each start must have fresh GPS and at least two seconds of stationary data remaining; the first target patch is entirely within the stop. The selected starts for this bundle are session1 0.32s, session2 9.28s, session3 0.32s, session4 0.32s, and session5 0.32s. All five start at measured speed zero. Each car follows the remaining chronological sessions, preserving recording pauses and omitting stale GPS rather than interpolating a path. There is no wraparound.

The seven output columns are `carID`, `timestamp`, `latitude`, `longitude`, `imu_defect_detected`, `yolo_pothole_detected`, and `road_quality`. Road quality is good/medium/bad. The IMU boolean uses the existing post-processed decision. A positive YOLO frame anywhere in the target patch makes the camera flag true; an observed negative interval makes it false. Missing video coverage is NULL, including session1. Invalid IMU predictions also remain NULL. Camera alignment is approximate, using the existing MP4 timestamp estimate.

Timestamp represents the target patch's road-observation time shifted to the synthetic scenario clock. These are offline, precomputed replay outputs, not a simulation of cloud latency or live causal delivery. GPS is the observer vehicle position, not a physical pothole triangulation. Multiple virtual cars copy the same underlying observations and must not count as independent evidence or training examples.

The generator only writes local files. To import later, `cd` into the bundle, set `DATABASE_URL` privately, and run `psql "$DATABASE_URL" -f load.sql`. The import is transactional and skips identical car/timestamp keys on retry. It writes to a separate `simulated_car_observations` table and leaves real inference and pothole tables untouched. CSV empty fields are SQL NULL; JSONL retains native booleans and null.

Validation: four unit tests cover missing speed, stops, video coverage, and exact cascade timing. A repeated generation produced byte-identical data files. All 65,306 rows were imported successfully into Tiger TEMPORARY tables, and retrying inserted zero rows; the transaction was rolled back. The validation transaction did not persist rows. The subsequent user-authorized import committed all 65,306 rows to `public.simulated_car_observations`, verified through a separate connection. `road_quality` is non-null TEXT restricted to lowercase good/medium/bad. See `database_validation.json` and `tiger_import_receipt.json` in the bundle.

The reusable Python importer is `scripts/import_fleet_scenarios.py BUNDLE_DIRECTORY` with `DATABASE_URL` in the environment. It verifies source hashes and labels, imports atomically, and rejects conflicting duplicate contents.
