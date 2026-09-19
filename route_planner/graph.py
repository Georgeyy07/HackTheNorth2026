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
import socket
from pathlib import Path
from typing import List, TypedDict

import urllib3.util.connection as _urllib3_connection

# This environment's outbound IPv6 is broken/blackholed: DNS returns IPv6
# addresses first for hosts like overpass-api.de, and urllib3 (used by both
# `requests` and osmnx) tries those first, hanging the full connect timeout
# on each before ever reaching the working IPv4 address -- confirmed
# directly, a routing request that should take ~15s took 100+ seconds this
# way. Forcing IPv4-only resolution process-wide fixes it for every HTTP
# call in this module (Overpass, Nominatim, Photon alike).
_urllib3_connection.allowed_gai_family = lambda: socket.AF_INET

import osmnx as ox
import requests

from alert_service.alert_math import EARTH_RADIUS_M, haversine_distance_m

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "road_graphs"
# Padding around the origin/destination bounding box, so a route can use a
# road that bulges slightly outside the straight-line box between the two
# points (e.g. going around a lake or highway interchange).
ROUTE_BBOX_BUFFER_M = 2000.0
# Cache bbox edges snap outward to this grid so nearby routes share a
# cached graph instead of each computing (and downloading) their own
# precise bbox. ~5.5km at the equator, coarser than any single test route.
GRID_SIZE_DEG = 0.05
# A mistyped or ambiguous address can geocode to the wrong side of the
# world (confirmed: this happened during testing -- Overpass reported the
# resulting bbox as "1,063 times" its normal query size and then hung for
# 180s before timing out). Reject far-apart pairs immediately instead of
# attempting to download a country-sized chunk of OpenStreetMap.
MAX_ROUTE_DISTANCE_M = 100_000.0  # 100km -- generous for a single city/region trip
# Fail fast rather than hanging on a slow/overloaded Overpass mirror.
ox.settings.requests_timeout = 30

# The default public Overpass instance is a shared, rate-limited resource
# that intermittently times out on the actual query endpoint even while
# responding fine on /api/status (observed directly: overpass-api.de's
# /api/status answered in ~1s while /api/interpreter connect-timed-out three
# times in a row, ~100s total). Fall back to other public mirrors rather
# than failing outright when that happens.
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://lz4.overpass-api.de/api",
]


def _download_graph_with_mirror_fallback(download_fn, attempts_per_mirror: int = 2):
    """Some connection failures are transient (a specific backend IP behind a
    round-robin DNS entry being down, rather than the whole mirror) --
    confirmed directly: curl succeeded against overpass-api.de within the
    same few seconds a Python retry against it failed. A couple of quick
    retries per mirror costs little and catches these without needing a
    fully custom connection-pooling/retry setup around osmnx's internal
    request call."""
    last_error = None
    for mirror in OVERPASS_MIRRORS:
        ox.settings.overpass_url = mirror
        for _ in range(attempts_per_mirror):
            try:
                return download_fn()
            except Exception as exc:  # noqa: BLE001 -- osmnx wraps several distinct network errors
                last_error = exc
    raise RuntimeError(f"All Overpass mirrors failed; last error: {last_error}") from last_error


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
        graph = _download_graph_with_mirror_fallback(lambda: ox.graph_from_place(place, network_type="drive"))
        had_travel_times = False

    graph = _ensure_travel_times(graph)
    if not had_travel_times:
        ox.save_graphml(graph, cache_path)
    return graph


def load_road_graph_for_route(
    origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float,
    cache_dir: Path = DEFAULT_CACHE_DIR, buffer_m: float = ROUTE_BBOX_BUFFER_M,
):
    """Returns a networkx MultiDiGraph covering the drivable roads between
    (origin_lat, origin_lon) and (dest_lat, dest_lon), downloaded and cached
    by bounding box -- works anywhere in the world, not just one hardcoded
    city (a fixed `place` name meant routing silently broke, or geocoded to
    the wrong nearest node, for any city other than the one hardcoded).
    Every edge carries `length` (meters), `speed_kph`, and `travel_time` (seconds).
    Raises ValueError if the two points are farther apart than
    MAX_ROUTE_DISTANCE_M, rather than attempting to download and route over
    a country-sized road network."""
    distance_m = haversine_distance_m(origin_lat, origin_lon, dest_lat, dest_lon)
    if distance_m > MAX_ROUTE_DISTANCE_M:
        raise ValueError(
            f"Origin and destination are {distance_m / 1000:.0f}km apart, over the "
            f"{MAX_ROUTE_DISTANCE_M / 1000:.0f}km limit -- check the addresses geocoded "
            f"to the right place, or pick two points closer together."
        )

    south, north = sorted((origin_lat, dest_lat))
    west, east = sorted((origin_lon, dest_lon))

    lat_buffer_deg = buffer_m / EARTH_RADIUS_M * (180 / math.pi)
    mean_lat = (south + north) / 2
    lon_buffer_deg = buffer_m / (EARTH_RADIUS_M * max(0.1, math.cos(math.radians(mean_lat)))) * (180 / math.pi)

    bbox = (west - lon_buffer_deg, south - lat_buffer_deg, east + lon_buffer_deg, north + lat_buffer_deg)
    # Snapped outward to a coarse grid (never shrinks the requested area) so
    # that a second route nearby -- a very common case: same demo, slightly
    # different address, or just retrying -- reuses the graph already on
    # disk instead of hitting the flaky Overpass connection again. Without
    # this, every unique origin/destination pair computed its own precise
    # bbox and missed the cache even when it substantially overlapped an
    # already-downloaded area (confirmed: three near-identical Waterloo
    # queries during testing produced three different cache files).
    bbox = (
        math.floor(bbox[0] / GRID_SIZE_DEG) * GRID_SIZE_DEG,
        math.floor(bbox[1] / GRID_SIZE_DEG) * GRID_SIZE_DEG,
        math.ceil(bbox[2] / GRID_SIZE_DEG) * GRID_SIZE_DEG,
        math.ceil(bbox[3] / GRID_SIZE_DEG) * GRID_SIZE_DEG,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = "bbox_" + "_".join(f"{v:.3f}" for v in bbox).replace(".", "p").replace("-", "n")
    cache_path = cache_dir / f"{cache_key}.graphml"

    if cache_path.is_file():
        graph = ox.load_graphml(cache_path)
        had_travel_times = any("travel_time" in data for _, _, data in graph.edges(data=True))
    else:
        graph = _download_graph_with_mirror_fallback(lambda: ox.graph_from_bbox(bbox, network_type="drive"))
        had_travel_times = False

    graph = _ensure_travel_times(graph)
    if not had_travel_times:
        ox.save_graphml(graph, cache_path)
    return graph


def geocode_address(address: str) -> tuple[float, float]:
    """Turns a human address/place string (e.g. "200 University Ave W, Waterloo, ON")
    into (lat, lon) via OpenStreetMap's Nominatim geocoder. Needs network access;
    raises ValueError if Nominatim can't find a match."""
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

    A plain linear scan with our own haversine math, rather than
    osmnx's KDTree-backed `nearest_nodes` (which needs scikit-learn) --
    fine at the few-thousand-node scale of a single city's road graph.
    """
    return min(
        graph.nodes,
        key=lambda n: haversine_distance_m(lat, lon, graph.nodes[n]["y"], graph.nodes[n]["x"]),
    )
