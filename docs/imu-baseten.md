# Original IMU ensemble on Baseten

Branch `feat/baseten-imu-inference` starts from main commit
`77700d70b6ce26a548956d048b9b713481e4b013`. It contains the original four
ordinal/disturbance models (seeds 52–55), not the rejected pseudo-label students.
The four portable checkpoints total about 12 MB and are included in
`models/ordinal_pvs`. Every checkpoint is checked against the original SHA-256
manifest before packaging or loading.

The live deployment is in the **Hack the North** Baseten team:

- [Model and deployment logs](https://app.baseten.co/models/qk54002q/logs/woovmmk)
- Pinned prediction URL: `https://model-qk54002q.api.baseten.co/deployment/woovmmk/predict`
- Model ID `qk54002q`; deployment ID `woovmmk`.
- Plain FP32 PyTorch 2.9.0 on CPU, without compilation or quantization.
- Baseten selected a 2-vCPU / 8-GiB instance. This deployment has one maximum
  replica and can scale to zero; the first request may take longer.

## Start FastAPI

From the repository on your Mac:

```bash
git fetch origin
git switch --track origin/feat/baseten-imu-inference
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[viewer,alerts,routing,inference]'
cp -n .env.example .env
```

Edit `.env`: set `BASETEN_API_KEY` and `IMU_DATABASE_URL` to the Tiger PostgreSQL
connection URL, retaining the provider's SSL parameters. `DATABASE_URL` or
`TIGER_DATA_URL` are also recognized if `IMU_DATABASE_URL` is unset.

For an explicitly local development database, use:

```dotenv
IMU_DATABASE_URL=sqlite:///artifacts/imu_predictions.db
```

Then run:

```bash
python -m uvicorn road_viewer.server:create_app --factory --host 0.0.0.0 --port 8000
```

The existing backend `.env` loader reads these settings. Environment variables
take precedence over `.env`. No key is committed or sent to the mobile client.
The backend fails startup if inference is enabled but its configuration is
incomplete. Set `IMU_INFERENCE_ENABLED=false` for the original acknowledgement-only
IMU endpoint.

Check `GET /api/imu/status` for the active calibration, checkpoint manifest hash,
filter settings, and actual storage backend. Setting a PostgreSQL URL never
silently switches inference storage to SQLite if the connection fails.

## Input and processing contract

The mobile hook now sends samples from `MotionPipeline`, using VQF earth-frame
acceleration and the mount's projected forward direction to reproduce the
recorded `vqf_vehicle` inputs. Acceleration retains gravity in m/s²; channels
are forward, left, up, and speed in m/s. Missing channels remain null, and zero
speed remains a valid observation. Gyroscope is used for orientation, not as
an extra model input.

The initial WebSocket message to `/ws/imu` is:

```json
{"type":"handshake","protocol":"roughroute.imu.v1","sample_rate_hz":100,"acceleration_frame":"vqf_vehicle","client":"ios"}
```

The reply supplies a new `session_id`. Each sample batch supplies an increasing
integer `batch_id` and 1–1024 samples:

```json
{
  "type": "imu_batch",
  "batch_id": 1,
  "samples": [{
    "time": 0.0,
    "available_at_ms": 1790000000000,
    "segment": 0,
    "accel_x": 0.02,
    "accel_y": -0.04,
    "accel_z": 9.81,
    "speed": 8.4,
    "latitude": 43.4723,
    "longitude": -80.5449,
    "gps_timestamp_ms": 1790000000000,
    "gps_received_at_ms": 1790000000000,
    "settling": false
  }]
}
```

`time` is sensor time in seconds on the pipeline's 100-Hz grid, not callback time.
`available_at_ms` is Unix milliseconds when the sample became available. The
mobile batch's old `rate_hz: 5` describes packet frequency, not the sample rate.
Samples with absent acceleration are masked; speed alone cannot make a patch
valid. Missing grid positions become masked samples. A sensor segment change or
a gap longer than 10 seconds requires a new handshake; the mobile queue starts
a new inference session at a segment boundary.

Processing follows the original implementation:

1. Roll a 1,024-sample (10.24-second) window every 16 samples (160 ms), with masked
   left padding at startup. Baseten averages the four members' quality
   probabilities and sigmoid disturbance probabilities. It returns the last
   three patch predictions per context.
2. FastAPI combines each patch's available votes from three successive contexts
   with chronological weights 1, 2, 3. The two newest patches stay provisional.
3. After consensus, the saved car calibration subtracts **2.7712255020116716**
   from each cumulative ordinal logit, once. This adjusts road quality only.
4. Finalized defect scores pass through the original probability-space Kalman
   filter (**Q/R = 3.2**) and hysteresis (**onset 0.7, offset 0.5**). Provisional
   updates do not advance it. Missing patches reset the filter and censor active
   events.

The algorithmic finalization delay is 320 ms beyond a target patch's end.
Network, queueing, model startup and database latency add to that delay. The
mobile stream keeps one batch in flight, batches accumulated samples, retries
the same unacknowledged batch, and stops if its bounded queue overflows.

`IMU_CALIBRATION_PROFILE=phone_same_car_20260920` explicitly opts into the saved
original car/phone-mount calibration used for sessions 1–4. Use `none` for another
car or mount until separately calibrated. These are good/medium/bad probabilities
and a disturbance score, not a numeric IRI measurement. Stopping the car does not
trigger a hard-coded bad-road label. VQF settling is retained as metadata.

## Persistence and responses

Startup creates only additive `imu_sessions` and `imu_predictions` tables plus
an observation-time index. It uses TIMESTAMPTZ and JSONB on PostgreSQL and does not need Timescale extensions. Existing
pothole tables are preserved; the previous automatic drop-on-schema-difference
behavior was removed. Automatic demo pothole seeding is limited to the local
legacy database.

Finalized patches from each request are inserted as one batch in a transaction.
Every finalized 160-ms patch is saved with:

- session and patch identity, observation and computation times;
- raw and calibrated quality probabilities and ordinal class;
- original and Kalman-filtered defect scores, hysteresis state and event transitions;
- checkpoint hash, calibration ID/offset, validity, vote count and settling status;
- GPS coordinates only if a fix was available in the target patch and at most
  three seconds old; otherwise null.

`observed_at` uses the first sample's availability timestamp as the UTC anchor
plus relative sensor time. It is approximate by the initial sensor delivery
latency. `computed_at` is the server's actual computation time. Raw scores and
all provenance are retained in the JSON `payload` column; frequently queried
fields also have dedicated columns. Tables are ordinary PostgreSQL tables
compatible with Tiger, not configured as Timescale hypertables.

`(session_id, target_patch)` is unique. Database writes are transactional. The
backend commits stream/filter state only after successful persistence and caches
the latest acknowledged batch response, so retrying it does not double-advance
the Kalman filter or add rows. A reconnect creates a new session; resuming or
deduplicating across reconnects is not implemented. The unfinalized two-patch tail
is not persisted at disconnect.

The WebSocket acknowledgement contains `updates`, `persisted`, `session_id`, and
sample counters. The mobile hook exposes `latestPrediction`. Read committed rows
with:

```text
GET /api/imu/sessions/{session_id}/predictions?after=-1&limit=1000
```

`after` is the last target patch already read; the maximum page size is 5,000.
Disturbances are stored as inference observations, not automatically inserted
as confirmed potholes into the existing routing/alert table. Multi-car spatial
reconciliation remains a separate concern.

## Repackage and deploy

```bash
python scripts/package_imu.py
pip install truss
truss auth login
truss push baseten/imu_ensemble --team 'Hack the North' --non-interactive --output json
```

The packaging script verifies all four original checkpoint hashes and copies
only the model code and weights into the Truss bundle. The generated bundle
folders are ignored by Git. Keep the new pinned deployment URL in the backend
configuration after redeploying. See the official [Truss push reference](https://docs.baseten.co/reference/cli/truss/push)
and [configuration reference](https://docs.baseten.co/development/model/configuration).

## Verification

```bash
pip install -e '.[test]'
pytest imu_inference/tests road_viewer/tests/test_imu_stream.py road_viewer/tests/test_camera_stream.py -q
node --test mobile/tests/*.test.mjs
```

Recorded evidence is in `baseten/verification.json`:

- Baseten vs local CPU output on a real session-1 window: maximum probability
  difference **1.79e-7**.
- Ported streaming vs the original source on 1,600 recorded samples with an
  inserted missing patch: **297 updates**, exact probability equality; quality
  grades, calibration, event transitions and Kalman behavior matched.
- FastAPI WebSocket → real Baseten → local SQLite: 320 recorded samples,
  **18 finalized rows**, calibrated, with an idempotent duplicate batch retry.
- FastAPI WebSocket → real Baseten → **live Tiger PostgreSQL over SSL**: 320
  recorded samples, **18 finalized rows** in test session
  `90a689bb-6c3c-4f2d-b3ef-6a7c5345064a`; rows were read back through the API and
  directly from PostgreSQL. Duplicate batch retry added no rows. Batched insert
  reduced this test request from 6.87 s to 3.63 s (individual observations, not a
  latency benchmark). A preceding smoke run left a separate 18-row test session
  `7413fce1-a9a7-49ac-bb47-3015fe74ec4f`. Both are marked
  `prepared-recording-smoke` in session metadata; existing potholes were untouched.
- Mobile transformation and motion tests pass; a physical phone streaming to
  this deployment has not yet been exercised.

To repeat the cloud/backend smoke test with an existing prepared recording:

```bash
python scripts/smoke_imu.py --samples /path/to/samples.parquet --start 3000 --count 320
```

This writes a new session to the configured database. It uses the recorded
acceleration/speed but assigns a fresh replay clock and no GPS coordinates.

The broader existing suite has two pre-existing severity-mapping failures, also
reproduced using `origin/main`'s database module:
`test_potholes_api_crud_operations` and
`test_query_nearby_maps_columns_and_severity_and_filters_by_radius`.
They are unrelated to the new IMU observation tables.
