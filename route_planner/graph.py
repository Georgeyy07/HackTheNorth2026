"""Loads a real drivable road network via OSMnx, caching it to disk so repeated
runs (and the demo) don't re-download from OpenStreetMap every time.

Every edge gets a real `travel_time` (seconds), imputed by osmnx from OSM's
`maxspeed` tags where present, and from the mean maxspeed of other edges of
the same road type where it's missing (missing tags are common -- most
residential streets have no explicit maxspeed in OSM). That's what
"efficiency" is measured in for routing, rather than raw distance, so a
highway edge and a residential edge of the same length aren't treated as
equally efficient.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, TypedDict

import osmnx as ox
import requests

from alert_service.alert_math import EARTH_RADIUS_M, haversine_distance_m

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "road_graphs"
# Padding around the origin/destination bounding box, so a route can use a
# road that bulges slightly outside the straight-line box between the two
# points (e.g. going around a lake or highway interchange).
ROUTE_BBOX_BUFFER_M = 2000.0


def _ensure_travel_times(graph):
    if not any("travel_time" in data for _, _, data in graph.edges(data=True)):
        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)
    return graph


def load_road_graph(place: str, cache_dir: Path = DEFAULT_CACHE_DIR):
    """Returns a networkx MultiDiGraph of the drivable road network for `place`
    (e.g. "Waterloo, Ontario, Canada"), downloading and caching it on first use.
    Every edge carries `length` (meters), `speed_kph`, and `travel_time` (seconds)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{place.replace(' ', '_').replace(',', '')}.graphml"

    if cache_path.is_file():
        graph = ox.load_graphml(cache_path)
        had_travel_times = any("travel_time" in data for _, _, data in graph.edges(data=True))
    else:
        graph = ox.graph_from_place(place, network_type="drive")
        had_travel_times = False

    graph = _ensure_travel_times(graph)
    if not had_travel_times:
        ox.save_graphml(graph, cache_path)
    return graph


def load_road_graph_for_route(
    origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float,
    cache_dir: Path = DEFAULT_CACHE_DIR, buffer_m: float = None,
):
    """Returns a networkx MultiDiGraph covering the drivable roads between
    (origin_lat, origin_lon) and (dest_lat, dest_lon), downloaded and cached
    by bounding box -- works anywhere in the world, not just one hardcoded
    city. Every edge carries `length` (meters), `speed_kph`, and `travel_time` (seconds)."""
    south, north = sorted((origin_lat, dest_lat))
    west, east = sorted((origin_lon, dest_lon))

    if buffer_m is None:
        approx_dist_m = haversine_distance_m(origin_lat, origin_lon, dest_lat, dest_lon)
        # Adapt buffer for long routes to avoid downloading gigantic bounding boxes
        if approx_dist_m > 20000:
            buffer_m = 1000.0
        elif approx_dist_m > 10000:
            buffer_m = 1400.0
        else:
            buffer_m = ROUTE_BBOX_BUFFER_M

    lat_buffer_deg = buffer_m / EARTH_RADIUS_M * (180 / math.pi)
    mean_lat = (south + north) / 2
    lon_buffer_deg = buffer_m / (EARTH_RADIUS_M * max(0.1, math.cos(math.radians(mean_lat)))) * (180 / math.pi)

    bbox = (west - lon_buffer_deg, south - lat_buffer_deg, east + lon_buffer_deg, north + lat_buffer_deg)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = "bbox_" + "_".join(f"{v:.3f}" for v in bbox).replace(".", "p").replace("-", "n")
    cache_path = cache_dir / f"{cache_key}.graphml"

    if cache_path.is_file():
        graph = ox.load_graphml(cache_path)
        had_travel_times = any("travel_time" in data for _, _, data in graph.edges(data=True))
    else:
        graph = ox.graph_from_bbox(bbox, network_type="drive")
        had_travel_times = False

    graph = _ensure_travel_times(graph)
    if not had_travel_times:
        ox.save_graphml(graph, cache_path)
    return graph


from functools import lru_cache

_KNOWN_PLACES = {
    "university of waterloo": (43.4723, -80.5449),
    "university of waterloo, waterloo, on": (43.4723, -80.5449),
    "waterloo public square": (43.4623, -80.5224),
    "waterloo public square, waterloo, on": (43.4623, -80.5224),
    "waterloo park": (43.4685, -80.5312),
    "laurier": (43.4738, -80.5275),
    "wilfrid laurier university": (43.4738, -80.5275),
}


@lru_cache(maxsize=256)
def geocode_address(address: str) -> tuple[float, float]:
    """Turns a human address/place string (e.g. "200 University Ave W, Waterloo, ON")
    into (lat, lon) via OpenStreetMap's Nominatim geocoder. Needs network access;
    raises ValueError if Nominatim can't find a match."""
    cleaned = address.strip()
    # Check if raw coordinate format "lat, lon"
    if "," in cleaned:
        parts = cleaned.split(",")
        if len(parts) == 2:
            try:
                return float(parts[0].strip()), float(parts[1].strip())
            except ValueError:
                pass

    lower = cleaned.lower()
    if lower in _KNOWN_PLACES:
        return _KNOWN_PLACES[lower]

    try:
        return ox.geocode(address)
    except Exception as exc:  # osmnx raises its own InsufficientResponseError etc.
        raise ValueError(f"Could not geocode address: {address!r}") from exc


class AddressSuggestion(TypedDict):
    display_name: str
    lat: float
    lon: float


_GEOCODER_USER_AGENT = "HackTheNorth2026-RoughRoute/1.0 (hackathon project; contact via GitHub)"


def _format_photon_name(properties: dict) -> str:
    parts = [properties.get(k) for k in ("name", "street", "city", "state", "country")]
    # Drop consecutive duplicates (e.g. name == city for a city-level result).
    deduped: List[str] = []
    for part in parts:
        if part and (not deduped or part != deduped[-1]):
            deduped.append(part)
    return ", ".join(deduped)


def suggest_addresses(query: str, limit: int = 5) -> List[AddressSuggestion]:
    """Returns up to `limit` candidate addresses/places matching a partial
    query, for autocomplete-as-you-type. Uses Photon (Komoot's free,
    OSM-data-backed geocoder built for exactly this) rather than Nominatim's
    /search -- Nominatim's usage policy explicitly forbids "search as you
    type"/autocomplete use, and its full-text search isn't prefix-matching
    anyway (querying "University of Wat" against Nominatim returns an
    unrelated Polish military academy, not Waterloo; Photon returns
    University of Waterloo as the top hit). Returns [] on any network error
    or empty result -- callers shouldn't treat "no suggestions yet" as an
    error worth surfacing to the user while they're mid-typing."""
    query = query.strip()
    if len(query) < 3:
        return []
    try:
        response = requests.get(
            "https://photon.komoot.io/api/",
            params={"q": query, "limit": limit},
            headers={"User-Agent": _GEOCODER_USER_AGENT},
            timeout=8,
        )
        response.raise_for_status()
        features = response.json().get("features", [])
    except requests.RequestException:
        return []

    return [
        {
            "display_name": _format_photon_name(f["properties"]),
            "lon": f["geometry"]["coordinates"][0],
            "lat": f["geometry"]["coordinates"][1],
        }
        for f in features
    ]


def nearest_node(graph, lat: float, lon: float):
    """Finds the graph node closest to a raw GPS coordinate.
    Uses fast local equirectangular degree projection instead of full haversine scan."""
    cos_lat = math.cos(math.radians(lat))
    best_node = None
    best_dist_sq = math.inf
    for n, data in graph.nodes.items():
        dy = lat - data["y"]
        dx = (lon - data["x"]) * cos_lat
        d_sq = dy * dy + dx * dx
        if d_sq < best_dist_sq:
            best_dist_sq = d_sq
            best_node = n
    return best_node
