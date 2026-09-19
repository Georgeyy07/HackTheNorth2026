"""Loads a real drivable road network via OSMnx, caching it to disk so repeated
runs (and the demo) don't re-download from OpenStreetMap every time.
"""

from __future__ import annotations

from pathlib import Path

import osmnx as ox

from alert_service.alert_math import haversine_distance_m

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "road_graphs"


def load_road_graph(place: str, cache_dir: Path = DEFAULT_CACHE_DIR):
    """Returns a networkx MultiDiGraph of the drivable road network for `place`
    (e.g. "Waterloo, Ontario, Canada"), downloading and caching it on first use."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{place.replace(' ', '_').replace(',', '')}.graphml"
    if cache_path.is_file():
        return ox.load_graphml(cache_path)

    graph = ox.graph_from_place(place, network_type="drive")
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
