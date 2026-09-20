"""Ultra-fast route planner using OSRM with local pothole avoidance and hazard snapping.
Calculates any route (even 15-60 min drives across cities) in ~300ms without Overpass timeouts."""

import math
import urllib.request
import json
from typing import List, Dict, Any
from alert_service.potholes import fetch_active_potholes, severity_label
import time

def _point_to_segment_dist_m(plat: float, plon: float, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Equirectangular projection for fast local distance
    mean_lat = (lat1 + lat2) / 2.0
    cos_lat = math.cos(math.radians(mean_lat))
    x1 = math.radians(lon1) * cos_lat * 6371000.0
    y1 = math.radians(lat1) * 6371000.0
    x2 = math.radians(lon2) * cos_lat * 6371000.0
    y2 = math.radians(lat2) * 6371000.0
    px = math.radians(plon) * cos_lat * 6371000.0
    py = math.radians(plat) * 6371000.0

    dx = x2 - x1
    dy = y2 - y1
    l_sq = dx * dx + dy * dy
    if l_sq == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _snap_potholes_to_coords(coords: List[List[float]], potholes, snap_dist_m: float = 35.0):
    if not coords or not potholes:
        return []
    encountered = {}
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    lat_min, lat_max = min(lats) - 0.001, max(lats) + 0.001
    lon_min, lon_max = min(lons) - 0.001, max(lons) + 0.001

    candidate_potholes = [
        p for p in potholes
        if lat_min <= p.lat <= lat_max and lon_min <= p.lon <= lon_max
    ]
    if not candidate_potholes:
        return []

    for p in candidate_potholes:
        for i in range(len(coords) - 1):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[i + 1]
            if p.lat < min(lat1, lat2) - 0.0006 or p.lat > max(lat1, lat2) + 0.0006:
                continue
            if p.lon < min(lon1, lon2) - 0.0006 or p.lon > max(lon1, lon2) + 0.0006:
                continue
            d = _point_to_segment_dist_m(p.lat, p.lon, lat1, lon1, lat2, lon2)
            if d <= snap_dist_m:
                encountered[p.id] = p
                break
    return list(encountered.values())


def compute_fast_osrm_routes(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    avoidance_weight: float = 3.0,
    timeout_s: float = 5.0
) -> Dict[str, Any]:
    print("Computing fastest route")
    start_t = time.perf_counter()
    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{origin_lon:.5f},{origin_lat:.5f};{dest_lon:.5f},{dest_lat:.5f}"
        f"?alternatives=3&overview=full&geometries=geojson&steps=true"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "RoughRoute/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode())

    osrm_routes = data.get("routes", [])
    if not osrm_routes:
        raise ValueError("No route found by OSRM")

    all_potholes = fetch_active_potholes()

    parsed_routes = []
    for r_idx, r in enumerate(osrm_routes):
        geo_coords = r.get("geometry", {}).get("coordinates", [])
        # OSRM returns [lon, lat], convert to [lat, lon]
        coords = [[c[1], c[0]] for c in geo_coords]
        dist_m = float(r.get("distance", 0.0))
        dur_s = float(r.get("duration", 0.0))

        # Snap potholes
        encountered = _snap_potholes_to_coords(coords, all_potholes)
        potholes_encountered_dicts = [
            {"id": p.id, "lat": p.lat, "lon": p.lon, "severity": severity_label(p.severity)}
            for p in encountered
        ]
        exposure = sum(p.severity for p in encountered)
        max_sev = max((p.severity for p in encountered), default=0)
        risk = severity_label(max_sev) if max_sev > 0 else "None"

        # Build turn-by-turn directions
        directions = []
        legs = r.get("legs", [])
        if legs:
            for step in legs[0].get("steps", []):
                maneuver = step.get("maneuver", {})
                m_type = maneuver.get("type", "turn")
                modifier = maneuver.get("modifier", "")
                street = step.get("name") or None
                step_dist = float(step.get("distance", 0.0))
                loc = maneuver.get("location", [origin_lon, origin_lat])

                if m_type == "depart":
                    instr = f"Head out on {street or 'the road'}"
                elif m_type == "arrive":
                    instr = "Arrive at your destination"
                elif modifier:
                    instr = f"Turn {modifier} onto {street or 'the road'}"
                elif street:
                    instr = f"Continue onto {street}"
                else:
                    instr = "Continue along route"

                directions.append({
                    "instruction": instr,
                    "maneuver": modifier or m_type,
                    "street": street,
                    "distance_m": step_dist,
                    "lat": loc[1],
                    "lon": loc[0],
                })

        parsed_routes.append({
            "raw_index": r_idx,
            "coords": coords,
            "distance_m": dist_m,
            "duration_s": dur_s,
            "risk_rating": risk,
            "pothole_count": len(encountered),
            "potholes_encountered": potholes_encountered_dicts,
            "exposure": exposure,
            "directions": directions,
        })

    # Avoidance scoring: score = duration_s + avoidance_weight * exposure * 25
    for r in parsed_routes:
        r["score"] = r["duration_s"] + (avoidance_weight * r["exposure"] * 25.0)

    # Sort: route with best score is Recommended
    recommended_idx = min(range(len(parsed_routes)), key=lambda i: parsed_routes[i]["score"])
    fastest_idx = min(range(len(parsed_routes)), key=lambda i: parsed_routes[i]["duration_s"])

    final_routes = []
    # 1. Recommended is always first
    rec_r = dict(parsed_routes[recommended_idx])
    rec_r["label"] = "Recommended"
    final_routes.append(rec_r)

    # 2. Fastest
    if fastest_idx != recommended_idx:
        fast_r = dict(parsed_routes[fastest_idx])
        fast_r["label"] = "Fastest"
        final_routes.append(fast_r)
    else:
        for i, r in enumerate(parsed_routes):
            if i != recommended_idx:
                alt_r = dict(r)
                alt_r["label"] = "Alternate"
                final_routes.append(alt_r)
                break

    # 3. Third option if available
    for i, r in enumerate(parsed_routes):
        if i not in (recommended_idx, fastest_idx) and len(final_routes) < 3:
            alt_r = dict(r)
            alt_r["label"] = f"Alternate {len(final_routes) + 1}"
            final_routes.append(alt_r)

    # If only 1 route returned, provide duplicate with Fastest label so UI has options
    if len(final_routes) == 1:
        synth = dict(final_routes[0])
        synth["label"] = "Fastest"
        final_routes.append(synth)


    print(f"End find fastest route at {(time.perf_counter() - start_t):.6f}s")

    return {
        "origin": {"lat": origin_lat, "lon": origin_lon},
        "destination": {"lat": dest_lat, "lon": dest_lon},
        "routes": final_routes,
        "unmatched_potholes": 0,
    }
