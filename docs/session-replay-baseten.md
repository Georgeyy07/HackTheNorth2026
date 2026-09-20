# Recorded sessions through Baseten

The presentation viewer accepts ordinal IMU predictions and sampled YOLO frames alongside the existing numeric IRI exports. Ordinal roughness is displayed as good/medium/bad and a relative 0–100 score (`50 * (p_medium + 2*p_bad)`), never as a measured IRI value.

`export_baseten_sessions.py` runs fresh cloud inference from the prepared VQF vehicle-frame sensor samples. It verifies the raw JSONL hash against the preparation receipt. It does not reuse the preparation's prediction files. The IMU path uses the original ensemble, 16-sample patches, three-context consensus, saved calibration and Kalman/hysteresis implementation in `imu_inference`. Finalized patches are also written to Tiger when `DATABASE_URL` is set. Tail patches remain provisional.

Supply `BASETEN_API_KEY` and optionally `DATABASE_URL` through the environment, then run with a Python environment containing the inference dependencies, pandas, pyarrow and OpenCV:

```bash
python scripts/export_baseten_sessions.py \
  --prepared /home/origami/pothole/reports/original_ensemble_restored_20260920/replay \
  --input /home/origami/pothole/inference_data \
  --output /home/origami/pothole/reports/baseten_sessions_presentation \
  --sessions session2 session3 session4 --yolo-fps 1
```

Cloud responses are cached by input hash within the export directory. Rerunning resumes successful calls. Do not carry these caches across deployment/checkpoint changes. Each completed session has a receipt containing deployment IDs, source hashes, counts and calibration details. A session appears in the completed manifest only after both inference paths finish successfully.

YOLO runs at the selected sampling rate, not at the original video's frame rate. Frame boxes use the deployment's 0.4 pothole confidence threshold. Video/IMU alignment uses MP4 creation time only when it is within five seconds of the recorded IMU origin; the viewer labels this as approximate. It is not a measured clock synchronization. Camera observations are exported for replay, but are not published as geolocated Tiger potholes because alignment is unverified. IMU and camera predictions remain separate.

Start the viewer against the completed export:

```bash
python -m road_viewer.server --host 0.0.0.0 --port 8770 \
  --export /home/origami/pothole/reports/baseten_sessions_presentation
```

Open http://localhost:8770. No API key is sent to the browser.

## Current execution status

On September 20, the provided API key successfully listed both deployments under **Hack the North**, but both were `INACTIVE` with zero replicas. Prediction requests returned HTTP 400 (deployment deactivated). Activation requests returned HTTP 400 with `VALIDATION_ERROR: You must add a payment method to deploy models.` No new cloud predictions were produced. Hackathon coverage must be resolved with Baseten or activation completed through the account before rerunning.

While blocked, port 8770 serves `reports/baseten_sessions_pending`, containing real sensor/GPS data for sessions 2–4 and empty prediction streams, with an explicit pending banner. It is deliberately separate from the completed cloud export. Session 1 is excluded at the user's request.

Validation: seven JS semantic/replay tests and six replay API tests pass. Browser checks loaded and scrubbed all three pending sessions without JS errors. The existing legacy pothole CRUD test still fails its severity-filter assertion; that path is unrelated to this change.
