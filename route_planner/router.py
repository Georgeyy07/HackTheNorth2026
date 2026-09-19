"""Computes several named route options (e.g. "Fastest", "Balanced", "Avoid potholes")
over a road graph, each carrying real ETA and the actual severity-rated potholes it
passes -- so a caller (or UI) can let the user pick between real alternatives
instead of being handed one silently-chosen route.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Tuple

import networkx as nx

from alert_service.alert_math import PotholeReport
from alert_service.potholes import severity_label

from .cost import ROUTE_COST_ATTR, EdgeKey, RoutingConfig, apply_edge_exposure_to_costs, snap_potholes_to_edges
from .directions import DirectionStep, build_directions

# (label, avoidance_weight) pairs computed by default. A route identical to an
# already-listed one (same road sequence) is skipped, so the caller may see
# fewer than len(DEFAULT_PRESETS) options back if, say, "Balanced" and "Avoid
# potholes" land on the same road when there's nothing nearby to avoid.
DEFAULT_PRESETS: List[Tuple[str, float]] = [
    ("Fastest", 0.0),
    ("Balanced", 2.0),
    ("Avoid potholes", 8.0),
]


@dataclass
class RouteResult:
    label: str
    nodes: List
    coords: List[Tuple[float, float]]
    distance_m: float
    duration_s: float
    pothole_exposure: float
    potholes_encountered: List[PotholeReport]
    risk_rating: str  # "None" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" -- worst pothole actually on this route
    directions: List[DirectionStep]


def _risk_rating(potholes: List[PotholeReport]) -> str:
    if not potholes:
        return "None"
    return severity_label(max(p.severity for p in potholes))


def _build_route_result(graph, label: str, nodes: List, edge_potholes: Dict[EdgeKey, List[PotholeReport]]) -> RouteResult:
    coords = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in nodes]
    distance_m = 0.0
    duration_s = 0.0
    encountered_by_id: Dict[str, PotholeReport] = {}

    for u, v in zip(nodes, nodes[1:]):
        candidates = graph.get_edge_data(u, v)
        best_key = min(candidates, key=lambda k: candidates[k].get("length", 0.0))
        edge_data = candidates[best_key]
        distance_m += edge_data.get("length", 0.0)
        duration_s += edge_data.get("travel_time", edge_data.get("length", 0.0))
        for pothole in edge_potholes.get((u, v, best_key), []):
            encountered_by_id[pothole.id] = pothole

    potholes_encountered = list(encountered_by_id.values())
    exposure = sum(p.severity for p in potholes_encountered)
    return RouteResult(
        label=label, nodes=nodes, coords=coords, distance_m=distance_m, duration_s=duration_s,
        pothole_exposure=exposure, potholes_encountered=potholes_encountered,
        risk_rating=_risk_rating(potholes_encountered),
        directions=build_directions(graph, nodes),
    )


def find_routes(
    graph,
    origin_node,
    destination_node,
    potholes: Iterable[PotholeReport],
    presets: List[Tuple[str, float]] = DEFAULT_PRESETS,
    config: RoutingConfig = RoutingConfig(),
) -> dict:
    """Returns {'routes': [RouteResult, ...], 'unmatched_potholes': [...]} --
    one RouteResult per preset in `presets` (skipping any that land on a road
    sequence identical to one already returned), each carrying real distance,
    ETA, and the actual PotholeReports it passes with a worst-case risk_rating.
    `unmatched_potholes` lists reports too far from any road to route around.
    """
    potholes = list(potholes)
    edge_potholes, unmatched = snap_potholes_to_edges(graph, potholes, config)

    results: List[RouteResult] = []
    seen_node_sequences = set()
    for label, avoidance_weight in presets:
        preset_config = replace(config, avoidance_weight=avoidance_weight)
        apply_edge_exposure_to_costs(graph, edge_potholes, preset_config)
        nodes = nx.shortest_path(graph, origin_node, destination_node, weight=ROUTE_COST_ATTR)

        node_sequence = tuple(nodes)
        if node_sequence in seen_node_sequences:
            continue
        seen_node_sequences.add(node_sequence)
        results.append(_build_route_result(graph, label, nodes, edge_potholes))

    return {"routes": results, "unmatched_potholes": unmatched}
