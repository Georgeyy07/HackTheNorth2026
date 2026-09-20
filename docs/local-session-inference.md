# Local session inference

Use the presentation branch to process the recordings without Baseten. The original IMU ensemble and its saved calibration/consensus/Kalman/hysteresis run unchanged. YOLO26 processes every video frame with the selected checkpoint and 0.4 pothole confidence threshold.

```bash
conda activate a
export YOLO_AUTOINSTALL=false
python scripts/infer_local_sessions.py \
  --input /home/origami/pothole/inference_data \
  --output /home/origami/pothole/reports/local_sessions_2_5 \
  --sessions session2 session3 session4 session5 --device cuda
python scripts/infer_local_video.py \
  --input /home/origami/pothole/inference_data \
  --output /home/origami/pothole/reports/local_sessions_2_5 \
  --checkpoint /home/origami/pothole/best.pt --batch 32
```

The video decoder must honor MP4 rotation metadata. These four recordings have 90-degree orientation tags, and this OpenCV build defaults automatic rotation off. The exporter explicitly enables it before decoding/inference. Detection boxes refer to the upright display image (1080×1920 for these recordings), and preview boxes are scaled to their associated JPEG dimensions. Timestamps come from decoded frame presentation times. Session receipts record rotation, source hashes, frame count and approximate video-to-IMU offset.

The output includes calibrated IMU tables in CSV/Parquet/JSONL, provisional and final update streams, disturbance events, a JSONL row for every video frame, and a flat CSV of all retained YOLO boxes. Empty video frames are explicitly represented. Consecutive detections are not interpreted as unique physical potholes. Review `README.md`, `validation.json` and `summary.json` in the output directory.

Ultralytics 8.4.155 was installed in environment `a` with `--no-deps`, plus missing inference dependencies (OpenCV 4.11.0.86, THOP 2.1.6, Polars 1.44.2/runtime and nvidia-ml-py). No existing package versions changed; PyTorch remains 2.9.0+cu128. The cloud platform SDK was deliberately omitted to preserve the existing httpx/FastAPI dependencies, so `pip check` notes that missing SDK. Local inference does not require it. Environment receipts and the CPU/GPU IMU smoke comparison are stored with the results.

The viewer itself uses the repository's existing `.venv`, which has the routing and FastAPI dependencies. Serve the finished export with:

```bash
.venv/bin/python -m road_viewer.server --host 0.0.0.0 --port 8770 \
  --export /home/origami/pothole/reports/local_sessions_2_5
```

Open http://localhost:8770. This viewer replays completed local results; it does not call Baseten. Camera/IMU timing is approximate (MP4 creation metadata), and IMU quality calibration for session5 assumes the same car/phone mounting as the original calibrated recordings.

Render annotated H.264 MP4s after both inference exports finish:

```bash
python scripts/render_inference_videos.py \
  --output /home/origami/pothole/reports/local_sessions_2_5 --workers 3
```

The renderer reads the saved frame detections without rerunning either model. It draws YOLO boxes/confidence and a separately labeled calibrated IMU grade/defect score, keeps source audio where present, and verifies encoded frame count and upright dimensions. Videos are linked from the camera panel and served at `/api/session/session2/annotated.mp4` (replace the session ID as needed; add `?download=true` to download).
