"""Browser replay with optional CSV uploads and background ensemble inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache
import gzip
import json
import os
import logging
from pathlib import Path
import threading
import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn


HERE = Path(__file__).resolve().parent
EXPORT = HERE.parent / "artifacts/test_drive_inference"
FILTERS = HERE.parent / "reports/alert_filter_20260919"
SIGNALS = ["time_s", "input_available_s", "accel_x", "accel_y", "accel_z",
           "gyro_x", "gyro_y", "gyro_z", "speed"]
UPDATE_FIELDS = ["target_patch", "start_s", "end_s", "available_s", "probability",
                 "disturbance", "iri_m_per_km", "quality_grade", "status", "is_final",
                 "valid", "event_id", "event_transition", "target_latitude_deg",
                 "target_longitude_deg", "target_gps_valid", "target_gps_fix_index",
                 "target_gps_age_s", "context_spread", "original_probability", "score_kind"]
UPDATE_FIELDS += ['quality_probability', 'quality_name']


def read(path):
    return json.loads(path.read_text())


def create_app(export=None, filters=None, *, uploads=None, ensemble=None, device='auto', infer_drive=None):
    export = Path(export or os.environ.get("ROAD_VIEWER_EXPORT", EXPORT)).resolve()
    filters = Path(filters or os.environ.get("ROAD_VIEWER_FILTERS", FILTERS)).resolve()
    if not (export / "manifest.json").is_file() and uploads is None:
        raise FileNotFoundError(f"No replay manifest in {export}. Set --export or ROAD_VIEWER_EXPORT.")
    manifest = read(export / "manifest.json") if (export/'manifest.json').is_file() else dict(sessions=[], samples=0, duration_s=0.)
    sessions = {s["session_id"]: s for s in manifest["sessions"]}
    folders = {s: export/s for s in sessions}
    upload_root = Path(uploads).resolve() if uploads is not None else None
    if upload_root:
        upload_root.mkdir(parents=True, exist_ok=True)
        for receipt in upload_root.glob('drive_*/manifest.json'):
            for meta in read(receipt)['sessions']:
                sessions[meta['session_id']] = meta
                folders[meta['session_id']] = receipt.parent/meta['session_id']
    executor = ThreadPoolExecutor(max_workers=1) if upload_root else None
    jobs, upload_lock = {}, threading.Lock()
    @asynccontextmanager
    async def lifespan(app):
        yield
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    profile_file = filters / "viewer_profiles.json"
    if manifest.get('score_filter_applied'):
        # The export already contains filtered updates. Old comparison profiles
        # belong to another model/export and must not silently replace them.
        config = manifest['alert_filter']
        label = 'Kalman + hysteresis' if config.get('kind') == 'kalman' else 'Filtered + hysteresis'
        profiles = [dict(id='original', label=label, config=config, applied_in_export=True)]
    else:
        profiles = read(profile_file)["profiles"] if profile_file.exists() else [dict(id="original", label="Original post-processing", config={})]
    profile_lookup = {p["id"]:p for p in profiles}
    app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=4)

    def session_folder(session_id):
        if session_id not in sessions:
            raise HTTPException(404, "Unknown recording")
        return folders[session_id]

    def updates_file(session_id, profile):
        folder = session_folder(session_id)
        if sessions[session_id].get('dataset') == 'user_csv' and profile != 'original':
            raise HTTPException(404, 'Uploaded drives use their own fixed Kalman filtering')
        if profile not in profile_lookup:
            raise HTTPException(404, "Unknown alert profile")
        return folder / "updates.jsonl.gz" if profile == "original" else filters / (profile+"_updates") / (session_id+".jsonl.gz")

    @lru_cache(maxsize=3)
    def payload(session_id, profile):
        folder = session_folder(session_id)
        update_path = updates_file(session_id, profile)
        meta = sessions[session_id]
        samples = pd.read_parquet(folder / "samples.parquet", columns=SIGNALS)
        gps = pd.read_parquet(folder / "gps_fixes.parquet")
        updates = []
        with gzip.open(update_path, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                updates.append({key: row.get(key) for key in UPDATE_FIELDS})
        # No finished event summaries or final sample predictions are supplied:
        # event state and scores must be reconstructed from available updates.
        signals = json.loads(samples.to_json(orient="split", index=False, double_precision=15))
        fixes = json.loads(gps[["time_s", "latitude_deg", "longitude_deg"]].to_json(orient="values", double_precision=15))
        selected_profile = profile_lookup[profile]
        if meta.get('dataset') == 'user_csv':
            selected_profile = dict(id='original', label='Kalman + hysteresis', config=meta['alert_filter'], applied_in_export=True)
        result = dict(session=meta, profile=selected_profile, signals=signals, gps=fixes, updates=updates,
                      duration_s=max(meta["duration_s"], max((u["available_s"] for u in updates), default=0.)),
                      sample_rate_hz=100, gps_max_age_s=3.)
        return json.dumps(result, allow_nan=False, separators=(",", ":")).encode()

    @app.get("/api/catalog")
    def catalog():
        available = list(sessions.values())
        return dict(sessions=available, total_samples=sum(s['samples'] for s in available),
                    duration_s=sum(s['duration_s'] for s in available), patch_ms=160, delay_ms=320,
                    uploads_enabled=upload_root is not None,
                    tile_url="https://tile.openstreetmap.org/{z}/{x}/{y}.png", profiles=profiles, default_profile="original")

    @app.get("/api/session/{session_id}")
    def session_data(session_id: str, profile: str = "original"):
        return Response(payload(session_id, profile), media_type="application/json",
                        headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/updates/{session_id}")
    def download_updates(session_id: str, profile: str = "original"):
        return FileResponse(updates_file(session_id, profile), filename=f"{session_id}.{profile}.jsonl.gz")

    @app.get("/health")
    def health():
        return dict(status="ok", sessions=len(sessions), samples=sum(s['samples'] for s in list(sessions.values())))

    if upload_root:
        def process(job_id, path, filename, options):
            try:
                jobs[job_id].update(status='processing', progress=0.)
                from road_training.infer_csv import export_drive, DEFAULT_ENSEMBLE
                import torch
                torch.set_num_threads(4)
                target = upload_root/job_id
                run = infer_drive or export_drive
                meta = run(path, target, session_id=job_id, ensemble=ensemble or DEFAULT_ENSEMBLE,
                           device=device, display_name=filename,
                           progress=lambda value: jobs[job_id].update(progress=value), **options)
                folders[job_id] = target/job_id
                sessions[job_id] = meta
                jobs[job_id].update(status='complete', progress=1., session_id=job_id)
            except Exception as error:
                logging.exception('CSV inference failed')
                jobs[job_id].update(status='failed', error=str(error) if isinstance(error, (ValueError, KeyError)) else 'Inference failed; see the server log.')
            finally:
                upload_lock.release()

        @app.post('/api/import')
        async def import_csv(request: Request, filename: str = 'drive.csv', time_unit: str = 's',
                             acceleration_unit: str = 'm/s2', speed_unit: str = 'm/s', columns: str = '{}'):
            try:
                mapping = json.loads(columns)
            except json.JSONDecodeError:
                raise HTTPException(400, 'Column mapping must be valid JSON')
            if time_unit not in ('s','ms','iso') or acceleration_unit not in ('m/s2','g') or speed_unit not in ('m/s','km/h'):
                raise HTTPException(400, 'Unsupported input unit')
            if not isinstance(mapping, dict):
                raise HTTPException(400, 'Column mapping must be a JSON object')
            if not upload_lock.acquire(blocking=False):
                raise HTTPException(409, 'Another recording is being processed; wait for it to finish')
            job_id = 'drive_'+uuid.uuid4().hex[:16]
            path = upload_root/(job_id+'.csv')
            try:
                size = 0
                with path.open('xb') as handle:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > 64*1024*1024:
                            raise HTTPException(413, 'CSV must be at most 64 MiB')
                        handle.write(chunk)
                if size == 0:
                    raise HTTPException(400, 'CSV is empty')
                jobs[job_id] = dict(status='queued', progress=0.)
                executor.submit(process, job_id, path, Path(filename).name[:200],
                    dict(columns=mapping, time_unit=time_unit, acceleration_unit=acceleration_unit, speed_unit=speed_unit))
            except BaseException:
                path.unlink(missing_ok=True)
                upload_lock.release()
                raise
            return dict(job_id=job_id)

        @app.get('/api/import/{job_id}')
        def import_status(job_id: str):
            if job_id not in jobs:
                raise HTTPException(404, 'Unknown import')
            return dict(jobs[job_id])

    @app.get("/export")
    def download():
        archive = export.with_suffix(".zip")
        if not archive.is_file():
            raise HTTPException(404, "Export archive is not available")
        return FileResponse(archive, filename=export.name + ".zip")

    @app.get("/")
    def index():
        return FileResponse(HERE / "static/index.html")

    @app.get("/favicon.ico", status_code=204)
    def favicon():
        return Response(status_code=204)

    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.mount("/vendor/leaflet", StaticFiles(directory=HERE / "node_modules/leaflet/dist", check_dir=False), name="leaflet")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--export", type=Path, help="Replay export directory containing manifest.json")
    parser.add_argument("--filters", type=Path, help="Optional alert profiles directory")
    parser.add_argument('--uploads', type=Path, help='Enable CSV uploads and store imported drives here; works without an existing export')
    parser.add_argument('--ensemble', type=Path, help='Ordinal ensemble receipt for CSV inference')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = parser.parse_args()
    if not (HERE / "node_modules/leaflet/dist").is_dir():
        parser.error("Leaflet is missing. Run npm ci --prefix road_viewer first.")
    uvicorn.run(create_app(args.export, args.filters, uploads=args.uploads, ensemble=args.ensemble, device=args.device), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
