"""Read-only browser replay of the frozen test-drive export."""
import argparse
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

import sys

# Ensure repository root is in sys.path when executed directly as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from road_viewer.tiger_db import init_db, seed_sample_potholes, get_potholes, add_pothole, update_pothole, delete_pothole
from alert_service.potholes import fetch_active_potholes, severity_label
from route_planner.cost import RoutingConfig
from route_planner.graph import geocode_address, load_road_graph_for_route, nearest_node
from route_planner.router import find_routes
_ROUTE_GRAPH_CACHE = {}


def _route_graph(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float):
    # Rounded to ~100m so re-queries for the same origin/destination (e.g. the
    # user just moving the pothole-caution slider and re-searching) reuse the
    # in-memory graph instead of reloading from disk each time.
    key = (round(origin_lat, 3), round(origin_lon, 3), round(dest_lat, 3), round(dest_lon, 3))
    if key not in _ROUTE_GRAPH_CACHE:
        _ROUTE_GRAPH_CACHE[key] = load_road_graph_for_route(origin_lat, origin_lon, dest_lat, dest_lon)
    return _ROUTE_GRAPH_CACHE[key]


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


def ensure_default_export(export_dir: Path):
    if (export_dir / "manifest.json").is_file():
        return
    export_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import pandas as pd
    sessions = []
    for name, count, lat, lng in [
        ('kaggle_fixture', 500, 43.4723, -80.5449),
        ('lira_fixture', 300, 39.6180, 22.4280)
    ]:
        folder = export_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({key: np.arange(count, dtype=float) / 100 for key in SIGNALS})
        if name.startswith('lira'):
            frame[['gyro_x', 'gyro_y', 'gyro_z']] = np.nan
        frame.to_parquet(folder / 'samples.parquet', index=False)
        pd.DataFrame({
            'time_s': [0.0, count / 200.0, count / 100.0],
            'latitude_deg': [lat, lat + 0.002, lat + 0.004],
            'longitude_deg': [lng, lng + 0.002, lng + 0.004]
        }).to_parquet(folder / 'gps_fixes.parquet', index=False)
        with gzip.open(folder / 'updates.jsonl.gz', 'wt') as handle:
            for patch in range(max(1, count // 16)):
                handle.write(json.dumps(dict(
                    target_patch=patch, start_s=patch*0.16, end_s=(patch+1)*0.16,
                    available_s=(patch+1)*0.16 + 0.32, probability=0.75 if patch % 5 == 0 else 0.1,
                    disturbance=patch % 5 == 0, is_final=True, iri_m_per_km=2.5,
                    target_latitude_deg=lat, target_longitude_deg=lng
                )) + '\n')
        sessions.append(dict(session_id=name, samples=count, duration_s=count / 100,
                             dataset=name.split('_')[0], timestamp_origin_unix_ns="1696680000000000000"))
    manifest = dict(sessions=sessions, samples=800, duration_s=8.0)
    (export_dir / 'manifest.json').write_text(json.dumps(manifest))


def create_app(export=None, filters=None):
    export_given = export is not None or "ROAD_VIEWER_EXPORT" in os.environ
    export = Path(export or os.environ.get("ROAD_VIEWER_EXPORT", EXPORT)).resolve()
    filters = Path(filters or os.environ.get("ROAD_VIEWER_FILTERS", FILTERS)).resolve()
    if not (export / "manifest.json").is_file():
        if export_given:
            raise FileNotFoundError(f"No replay manifest in {export}. Set --export or ROAD_VIEWER_EXPORT.")
        ensure_default_export(export)
    manifest = read(export / "manifest.json")
    sessions = {s["session_id"]: s for s in manifest["sessions"]}
    profile_file = filters / "viewer_profiles.json"
    profiles = read(profile_file)["profiles"] if profile_file.exists() else [dict(id="original", label="Original post-processing", config={})]
    profile_lookup = {p["id"]:p for p in profiles}
    app = FastAPI(docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=4)

    @app.middleware("http")
    async def no_cache_for_static(request, call_next):
        # Dev convenience: static/index.html edits should always show up on the
        # next reload, not get served from a stale conditionally-cached copy.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store"
        return response

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

    # Initialize Tiger Data database
    try:
        init_db()
        seed_sample_potholes()
    except Exception as e:
        pass

    @app.get("/api/catalog")
    def catalog():
        return dict(sessions=list(sessions.values()), total_samples=manifest["samples"],
                    duration_s=manifest["duration_s"], patch_ms=160, delay_ms=320,
                    tile_url="https://tile.openstreetmap.org/{z}/{x}/{y}.png", profiles=profiles, default_profile="original")

    @app.get("/api/potholes")
    def list_potholes(severity: str = None):
        return get_potholes(severity=severity)

    @app.post("/api/potholes")
    def create_pothole(payload: dict = Body(...)):
        if "latitude" not in payload or "longitude" not in payload:
            raise HTTPException(400, "Latitude and Longitude are required")
        new_record = add_pothole(
            latitude=float(payload["latitude"]),
            longitude=float(payload["longitude"]),
            severity=str(payload.get("severity", "MEDIUM"))
        )
        return new_record

    @app.put("/api/potholes/{pothole_id}")
    def modify_pothole(pothole_id: int, payload: dict = Body(...)):
        success = update_pothole(
            pothole_id=pothole_id,
            severity=payload.get("severity"),
            latitude=payload.get("latitude"),
            longitude=payload.get("longitude")
        )
        if not success:
            raise HTTPException(404, "Pothole not found or no changes made")
        return {"status": "ok", "id": pothole_id}

    @app.delete("/api/potholes/{pothole_id}")
    def remove_pothole(pothole_id: int):
        success = delete_pothole(pothole_id)
        if not success:
            raise HTTPException(404, "Pothole not found")
        return {"status": "ok", "deleted_id": pothole_id}

    @app.post("/api/potholes/seed")
    def seed_potholes():
        seed_sample_potholes()
        return {"status": "ok", "potholes": get_potholes()}

    @app.get("/api/route")
    def compute_route(origin: str, destination: str, avoidance_weight: float = 3.0):
        try:
            origin_lat, origin_lon = geocode_address(origin)
            dest_lat, dest_lon = geocode_address(destination)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

        graph = _route_graph(origin_lat, origin_lon, dest_lat, dest_lon)
        origin_node = nearest_node(graph, origin_lat, origin_lon)
        dest_node = nearest_node(graph, dest_lat, dest_lon)

        # Potholes are matched against roads in the loaded bbox, so anything
        # outside the origin/destination area is unmatched by definition --
        # not "too far from any road," just outside the region this route
        # cares about. Pre-filter to the graph's own bounding box first.
        graph_lats = [data["y"] for _, data in graph.nodes(data=True)]
        graph_lons = [data["x"] for _, data in graph.nodes(data=True)]
        lat_min, lat_max = min(graph_lats), max(graph_lats)
        lon_min, lon_max = min(graph_lons), max(graph_lons)
        potholes = [
            p for p in fetch_active_potholes()
            if lat_min <= p.lat <= lat_max and lon_min <= p.lon <= lon_max
        ]

        # "Fastest" (0) and "Max avoidance" (15, a strong fixed ceiling) stay as
        # reference points; "Recommended" is exactly the caller's dial, so
        # turning it up/down directly changes how hard that middle option
        # avoids potholes vs. chasing ETA. All three collapse toward "Fastest"
        # as avoidance_weight -> 0, at which point find_routes backfills real
        # alternate routes instead of returning duplicates.
        presets = [("Fastest", 0.0), ("Recommended", avoidance_weight), ("Max avoidance", 15.0)]

        try:
            result = find_routes(graph, origin_node, dest_node, potholes, presets=presets, config=RoutingConfig())
        except Exception as exc:
            raise HTTPException(400, f"No route found: {exc}")

        def serialize(route):
            return dict(
                label=route.label, coords=route.coords, distance_m=route.distance_m,
                duration_s=route.duration_s, risk_rating=route.risk_rating,
                pothole_count=len(route.potholes_encountered),
                potholes_encountered=[
                    dict(id=p.id, lat=p.lat, lon=p.lon, severity=severity_label(p.severity))
                    for p in route.potholes_encountered
                ],
                directions=[
                    dict(instruction=s.instruction, maneuver=s.maneuver, street=s.street,
                         distance_m=s.distance_m, lat=s.lat, lon=s.lon)
                    for s in route.directions
                ],
            )

        return dict(
            origin=dict(lat=origin_lat, lon=origin_lon),
            destination=dict(lat=dest_lat, lon=dest_lon),
            routes=[serialize(r) for r in result["routes"]],
            unmatched_potholes=len(result["unmatched_potholes"]),
        )

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
        return FileResponse(HERE / "static/index.html", headers={"Cache-Control": "no-store"})

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
