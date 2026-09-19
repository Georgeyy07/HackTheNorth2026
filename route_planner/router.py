"""Computes both a pure-efficiency route and a pothole-aware route over a road graph,
so a caller (or demo UI) can show the tradeoff explicitly rather than picking silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx

from alert_service.alert_math import PotholeReport

from .cost import ROUTE_COST_ATTR, EdgeKey, RoutingConfig, apply_edge_exposure_to_costs, snap_potholes_to_edges


@dataclass
class RouteResult:
    nodes: List
    coords: List[Tuple[float, float]]
    distance_m: float
    pothole_exposure: float


def _build_route_result(graph, nodes: List, edge_exposure: Dict[EdgeKey, float]) -> RouteResult:
    coords = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in nodes]
    distance_m = 0.0
    exposure = 0.0
    for u, v in zip(nodes, nodes[1:]):
        candidates = graph.get_edge_data(u, v)
        best_key = min(candidates, key=lambda k: candidates[k].get("length", 0.0))
        edge_data = candidates[best_key]
        distance_m += edge_data.get("length", 0.0)
        exposure += edge_exposure.get((u, v, best_key), 0.0)
    return RouteResult(nodes=nodes, coords=coords, distance_m=distance_m, pothole_exposure=exposure)


def find_route(
    graph,
    origin_node,
    destination_node,
    potholes: Iterable[PotholeReport],
    config: RoutingConfig = RoutingConfig(),
) -> dict:
    """Returns {'efficient': RouteResult, 'pothole_aware': RouteResult, 'unmatched_potholes': [...]}
    for the same origin/destination, so the two routes can be compared or shown side by side.
    `unmatched_potholes` lists reports too far from any road to route around (likely bad GPS)."""
    potholes = list(potholes)
    edge_exposure, unmatched = snap_potholes_to_edges(graph, potholes, config)
    apply_edge_exposure_to_costs(graph, edge_exposure, config)

    efficient_nodes = nx.shortest_path(graph, origin_node, destination_node, weight="length")
    aware_nodes = nx.shortest_path(graph, origin_node, destination_node, weight=ROUTE_COST_ATTR)

    return {
        "efficient": _build_route_result(graph, efficient_nodes, edge_exposure),
        "pothole_aware": _build_route_result(graph, aware_nodes, edge_exposure),
        "unmatched_potholes": unmatched,
    }
