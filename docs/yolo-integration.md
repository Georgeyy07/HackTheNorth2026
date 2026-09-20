# YOLO26 camera integration

This ports the inference logic from `adding-dockerfile` commit
`13cddc31df85611e17e93b9c518112be71c420c6` onto
`feat/baseten-imu-inference`. It keeps the existing IMU model, calibration,
consensus, Kalman filter and IMU database tables intact.

The source branch provided local Ultralytics inference, per-class thresholds,
a FastAPI fallback service, a detection-reporting HTTP client and Docker files.
It did not contain a Baseten prediction client or wire camera frames into the
main backend. Those connections are implemented here.

## Model provenance

`models/yolo26/best.pt` is checksum-bound by `vision_inference/manifest.json`.
Its SHA-256 is
`d5c7326eb9274d663b94e955bd373854a7b175c4a746df23a761c3f1687c7445`.
The deployed checkpoint is the user-selected file
`/home/origami/pothole/best.pt`, copied byte-for-byte into the repository and
verified before packaging. It also exactly matches the earlier branch artifact
and Baseten training job **32oo70w** in project **wx42dyq**. The file contains a
YOLO26n detector trained with Ultralytics 8.4.155 and **one class: pothole**.
The deployment uses this supplied file directly; no training-run recovery is required.

The production Dedicated Inference deployment is under **Hack the North**:

- Model `31l2451q`, `roughroute-yolo26-potholes`.
- [Deployment logs](https://app.baseten.co/models/31l2451q/logs/w60x715).
- URL `https://model-31l2451q.api.baseten.co/deployment/w60x715/predict`.
- Ordinary PyTorch CPU inference; source weights are unchanged.

## Configure the combined FastAPI backend

Install the existing backend extras plus `vision`:

```bash
pip install -e '.[viewer,alerts,routing,inference,vision]'
```

Keep the IMU settings from [imu-baseten.md](imu-baseten.md), and set:

```dotenv
YOLO_INFERENCE_ENABLED=true
YOLO_BASETEN_PREDICT_URL=https://model-31l2451q.api.baseten.co/deployment/w60x715/predict
YOLO_PUBLISH_POTHOLES=true
```

Use the existing `BASETEN_API_KEY` or a separate `YOLO_BASETEN_API_KEY`.
`VISION_DATABASE_URL` overrides the shared `DATABASE_URL` / `TIGER_DATA_URL` /
`IMU_DATABASE_URL` / `POSTGRES_URL`. Database credentials remain server-side.
An unavailable PostgreSQL connection never silently falls back to SQLite.
For explicit local development use `sqlite:///artifacts/vision.db`.

Start with `python -m uvicorn road_viewer.server:create_app --factory --port 8000`.
`GET /api/vision/status` reports the model manifest and storage configuration.
YOLO and IMU enablement are independent; neither invokes the other's model.
The root Dockerfile builds this combined backend, including its route-planner,
IMU and vision modules, without installing model runtimes inside the API container:

```bash
docker build -t roughroute-backend .
docker run --env-file .env -p 8000:8080 roughroute-backend
```

## Camera and HTTP contract

`POST /api/vision/predict`, `/ws/camera`, and camera packets on `/ws/imu` share the
same inference and persistence path. A frame contains:

```json
{
  "type": "camera_frame",
  "frame_id": "785b90cb-6c22-4b58-a51b-a66d3bcbad15",
  "device_id": "phone-camera-session",
  "frame_number": 1,
  "timestamp": 1790000000000,
  "image": "<base64 JPEG or PNG>",
  "latitude": 43.4723,
  "longitude": -80.5449,
  "gps_timestamp_ms": 1790000000000,
  "gps_received_at_ms": 1790000000000
}
```

Images are limited to 4 MiB and 16 megapixels. The camera now captures a fresh
image before sending it, attaches capture-time GPS, and allows one outstanding
request. It retries the same frame ID instead of repeatedly sending a cached
image as new observations. `targetFps` is a capture ceiling; actual throughput
is limited by inference and persistence acknowledgements. Images are oriented
through the camera's normal processing. The hook exposes `latestPrediction`.

The cloud receives only the base64 image. Detection responses use the source's
`detections: [{box: [x1,y1,x2,y2], conf, cls}]` contract, with pixel coordinates.
The ported model adds class labels, checkpoint hash and `stub: false`.
Backend validation rejects synthetic stubs, malformed/out-of-image boxes,
unexpected class names and mismatched checkpoint hashes. Thresholds retain the
source defaults: pothole **0.4**, crack **0.2** when a crack-capable checkpoint
is explicitly selected. A legacy external endpoint without a hash is supported;
its response is recorded without a verified checkpoint hash.

Acknowledgements contain the frame ID, detections, model/provider provenance,
persisted detection count, GPS and associated pothole IDs. Read a committed
response using `GET /api/vision/frames/{frame_id}`. A retry with the same frame ID
and identical request returns its saved response; changing the image or metadata
under an existing ID is rejected. Cloud or database failures are not successful
empty detections.

## Tiger writes and map reporting

The new additive tables are `vision_frames` and `vision_detections`. Every
processed frame, including a genuine empty result, has an idempotent frame record.
Detection rows preserve class, confidence, pixel bounding box, location and an
optional link to the existing `potholes` table. Images are not stored; only their
SHA-256 hashes are retained.

With `YOLO_PUBLISH_POTHOLES=true`, valid pothole detections with a fresh causal GPS
fix also update Tiger's existing multi-sensor potholes schema. Writes include
`confidence`, `visual_box_confidence`, `detected_by_vision`, `device_id`, and
`num_visual_detections`. Physical `severity` remains unknown (NULL): confidence
is not pothole depth or roughness. The existing alert reader handles NULL
severity with its neutral 0.5 routing score rather than crashing.

Reports from the same device within 8 m and 10 seconds share a map report to
limit repeated-frame duplicates; raw detections remain separate. Multiple
pothole boxes in one frame share that frame's map report. This is approximate
report grouping, not object tracking or multi-view triangulation. Coordinates
are the camera/vehicle position, not a measured pothole position. A fix must have
arrived by capture time and be at most three seconds old. Without that fix,
observations are saved with null coordinates and no map pothole is created.
Cracks stay in vision observations and are not mislabeled as potholes.

Frames, detections and map changes commit in one transaction. Retry handling
survives reconnects/process restarts as long as the client retains the same UUID.
The mobile camera retains a pending frame only while its streaming session is
active; it does not persist an offline queue across an app restart. Set
`YOLO_PUBLISH_POTHOLES=false` for observation-only operation. Existing Tiger
pothole tables are never dropped or automatically reshaped.

## Optional fallback service

The source fallback was ported into `yolo_fallback/server.py`, reusing the same
model adapter and thresholds as Baseten. It accepts multipart `file` uploads at
`POST /predict` and exposes `/health`. Missing or mismatched weights fail startup;
it never returns the source branch's fake detection when weights are missing.

```bash
pip install -r yolo_fallback/requirements.txt
python -m uvicorn yolo_fallback.server:create_app --factory --port 8081
# Or build from the repository root:
docker build -f yolo_fallback/Dockerfile -t roughroute-yolo-fallback .
```

Configure `YOLO_FALLBACK_URL=http://127.0.0.1:8081/predict` in the backend to use
it if Baseten fails. The Baseten authorization header is never forwarded to the
fallback. Sentry is optional via `SENTRY_DSN`; it is disabled when unset.
No Cloud Run fallback deployment is assumed or provisioned by this port.

## Import an exact training run and deploy

In a YOLO runtime environment with `BASETEN_API_KEY` set:

```bash
python scripts/import_yolo_checkpoint.py --project wx42dyq --job <selected-job-id>
python scripts/package_yolo.py
truss push baseten/yolo26 --team 'Hack the North' --promote --non-interactive --output json
```

The import script fails if the requested run has no unique best.pt; it never
chooses a different run automatically. It downloads that artifact, inspects its
class names and updates the manifest hash. Update `YOLO_BASETEN_PREDICT_URL` to
the new pinned deployment URL, and restart the backend after a manifest change.
The package script does not copy credentials or videos into the Truss bundle.

Primary references: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/),
[Baseten training job checkpoints](https://docs.baseten.co/reference/training-api/get-training-job-checkpoints),
and [Expo SDK 57](https://docs.expo.dev/versions/v57.0.0/).

## Validation

```bash
pytest vision_inference/tests imu_inference/tests road_viewer/tests/test_camera_stream.py road_viewer/tests/test_imu_stream.py -q
node --test mobile/tests/*.test.mjs
python scripts/smoke_vision.py --image /path/to/real-frame.jpg
```

The smoke command writes one frame to the configured database without GPS, so it
cannot create a test pothole on the real map. It verifies an HTTP inference,
read-back and a WebSocket retry against the same saved frame. The optional
PostgreSQL test uses temporary copies of Tiger's schema and rolls back all data;
set `TEST_VISION_POSTGRES_URL` to run it. It checks real PostgreSQL reporting,
JSONB records, null severity, duplicate retries and nearby-report grouping.

Recorded checks are in `baseten/yolo26/verification.json`: 27 Python tests and
33 mobile tests passed, plus the separate real PostgreSQL temporary-table test.
The actual cloud/backend/Tiger smoke run returned zero detections for its sample
image and verified persistence of that empty frame and retry identity. This
validates integration, not detection accuracy. Fallback inference was also
exercised locally with the real weights. The built Python wheel contains both
model manifests and IMU calibration. Docker image builds and a physical phone
stream were not exercised in this environment.
