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

from pathlib import Path

import osmnx as ox

from alert_service.alert_math import haversine_distance_m

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "road_graphs"


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
