"""Read-only browser replay of the frozen test-drive export."""
import argparse
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn


HERE = Path(__file__).resolve().parent
EXPORT = HERE.parent / "reports/test_drive_inference_20260919"
FILTERS = HERE.parent / "reports/alert_filter_20260919"
SIGNALS = ["time_s", "input_available_s", "accel_x", "accel_y", "accel_z",
           "gyro_x", "gyro_y", "gyro_z", "speed"]
UPDATE_FIELDS = ["target_patch", "start_s", "end_s", "available_s", "probability",
                 "disturbance", "iri_m_per_km", "quality_grade", "status", "is_final",
                 "valid", "event_id", "event_transition", "target_latitude_deg",
                 "target_longitude_deg", "target_gps_valid", "target_gps_fix_index",
                 "target_gps_age_s", "context_spread", "original_probability", "score_kind"]


def read(path):
    return json.loads(path.read_text())


def create_app(export=None, filters=None):
    export = Path(export or os.environ.get("ROAD_VIEWER_EXPORT", EXPORT)).resolve()
    filters = Path(filters or os.environ.get("ROAD_VIEWER_FILTERS", FILTERS)).resolve()
    if not (export / "manifest.json").is_file():
        raise FileNotFoundError(f"No replay manifest in {export}. Set --export or ROAD_VIEWER_EXPORT.")
    manifest = read(export / "manifest.json")
    sessions = {s["session_id"]: s for s in manifest["sessions"]}
    profile_file = filters / "viewer_profiles.json"
    profiles = read(profile_file)["profiles"] if profile_file.exists() else [dict(id="original", label="Original post-processing", config={})]
    profile_lookup = {p["id"]:p for p in profiles}
    app = FastAPI(docs_url=None, redoc_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=4)

    def session_folder(session_id):
        if session_id not in sessions:
            raise HTTPException(404, "Unknown recording")
        return export / session_id

    def updates_file(session_id, profile):
        folder = session_folder(session_id)
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
        result = dict(session=meta, profile=profile_lookup[profile], signals=signals, gps=fixes, updates=updates,
                      duration_s=max(meta["duration_s"], max((u["available_s"] for u in updates), default=0.)),
                      sample_rate_hz=100, gps_max_age_s=3.)
        return json.dumps(result, allow_nan=False, separators=(",", ":")).encode()

    @app.get("/api/catalog")
    def catalog():
        return dict(sessions=list(sessions.values()), total_samples=manifest["samples"],
                    duration_s=manifest["duration_s"], patch_ms=160, delay_ms=320,
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
        return dict(status="ok", sessions=len(sessions), samples=manifest["samples"])

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
    args = parser.parse_args()
    if not (HERE / "node_modules/leaflet/dist").is_dir():
        parser.error("Leaflet is missing. Run npm ci --prefix road_viewer first.")
    uvicorn.run(create_app(args.export, args.filters), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
