"""Turns raw pothole reports into a routing cost that trades off against real travel time.

Real phone GPS is noisy: a pothole report can land 100+ meters from any real
road (confirmed against the real Waterloo graph -- a real reported pothole
sat 178m from the *closest* road edge in the entire city). A fixed
distance-to-edge cutoff either misses noisy reports like that (buffer too
tight) or double-counts a pothole near a fork onto multiple edges (buffer
too loose). Instead, each pothole is snapped to its single nearest road edge
(the standard GPS map-matching approach) -- and if even the nearest edge is
farther than `max_snap_distance_m`, the report is treated as unmatched
(probably a bad GPS fix) rather than silently misapplied to the wrong road.

"Efficiency" is measured in real travel time (`travel_time`, seconds -- see
graph.py, which imputes it from OSM's maxspeed tags), not raw distance, so a
highway edge and an equally-long residential edge aren't treated as equally
efficient. Snapped exposure is scaled by `penalty_per_severity_s` (an
equivalent "extra seconds" cost per severity unit) and dialed up or down with
`avoidance_weight` -- 0 ignores potholes entirely (pure fastest-route), 1
applies the full penalty, and values above 1 push harder toward avoidance.

This still approximates each edge as a straight line rather than using OSM's
true polyline geometry (the 'geometry' attribute, when present) -- fine for
most segments, but a sharply curved road could still snap incorrectly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from alert_service.alert_math import EARTH_RADIUS_M, PotholeReport

ROUTE_COST_ATTR = "route_cost"
EdgeKey = Tuple[object, object, object]


@dataclass
class RoutingConfig:
    max_snap_distance_m: float = 150.0     # beyond this, a pothole is "unmatched" to any road
    penalty_per_severity_s: float = 30.0   # "extra seconds" added per unit severity, before avoidance_weight
    avoidance_weight: float = 0.5          # 0 = pure fastest route, 1 = full penalty, >1 = stronger avoidance


def _to_local_meters(lat: float, lon: float, ref_lat_deg: float) -> Tuple[float, float]:
    """Equirectangular projection to local (x, y) meters, accurate enough at
    single-city scale, centered near `ref_lat_deg` to keep the cos() correction local."""
    x = math.radians(lon) * math.cos(math.radians(ref_lat_deg)) * EARTH_RADIUS_M
    y = math.radians(lat) * EARTH_RADIUS_M
    return x, y


def _point_to_segment_distance_m(
    plat: float, plon: float, alat: float, alon: float, blat: float, blon: float
) -> float:
    """Shortest distance from point P to the line segment A-B, in meters."""
    ref_lat = (alat + blat) / 2
    px, py = _to_local_meters(plat, plon, ref_lat)
    ax, ay = _to_local_meters(alat, alon, ref_lat)
    bx, by = _to_local_meters(blat, blon, ref_lat)

    abx, aby = bx - ax, by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def pothole_exposure(lat: float, lon: float, potholes: Iterable[PotholeReport], buffer_m: float) -> float:
    """Sum of severities for potholes within `buffer_m` of a single point (lat, lon)."""
    return sum(
        p.severity
        for p in potholes
        if _point_to_segment_distance_m(p.lat, p.lon, lat, lon, lat, lon) <= buffer_m
    )


def snap_potholes_to_edges(
    graph, potholes: Iterable[PotholeReport], config: RoutingConfig = RoutingConfig()
) -> Tuple[Dict[EdgeKey, List[PotholeReport]], List[PotholeReport]]:
    """Assigns each pothole to its single nearest road edge.

    Returns (edge_potholes, unmatched) where edge_potholes maps (u, v, key) ->
    the list of PotholeReports snapped there (so callers can show which real
    potholes/severities a route passes, not just a summed number), and
    unmatched lists potholes farther than `max_snap_distance_m` from every
    edge -- likely bad GPS fixes, surfaced rather than silently dropped or
    misapplied.
    """
    edges = list(graph.edges(keys=True))
    nodes = graph.nodes
    edge_potholes: Dict[EdgeKey, List[PotholeReport]] = {}
    unmatched: List[PotholeReport] = []
    # ~150m cutoff in degrees (~0.0016 deg) for fast spatial pruning
    max_snap_deg = (config.max_snap_distance_m + 30.0) / 111000.0

    for pothole in potholes:
        p_lat, p_lon = pothole.lat, pothole.lon
        best_edge = None
        best_distance = math.inf
        for u, v, key in edges:
            uy, ux = nodes[u]["y"], nodes[u]["x"]
            vy, vx = nodes[v]["y"], nodes[v]["x"]
            if p_lat < min(uy, vy) - max_snap_deg or p_lat > max(uy, vy) + max_snap_deg:
                continue
            if p_lon < min(ux, vx) - max_snap_deg or p_lon > max(ux, vx) + max_snap_deg:
                continue
            d = _point_to_segment_distance_m(p_lat, p_lon, uy, ux, vy, vx)
            if d < best_distance:
                best_distance, best_edge = d, (u, v, key)

        if best_edge is None or best_distance > config.max_snap_distance_m:
            unmatched.append(pothole)
        else:
            edge_potholes.setdefault(best_edge, []).append(pothole)
            # A two-way street is stored as two separate directed edges with
            # near-identical geometry. Without this, a pothole snaps to
            # whichever direction happens to be enumerated first and is
            # invisible to routes traveling the other way down the same
            # physical road (confirmed against the real Waterloo graph).
            u, v, key = best_edge
            reverse_edge = (v, u, key)
            if graph.has_edge(v, u, key):
                edge_potholes.setdefault(reverse_edge, []).append(pothole)

    return edge_potholes, unmatched


def apply_edge_exposure_to_costs(
    graph, edge_potholes: Dict[EdgeKey, List[PotholeReport]], config: RoutingConfig = RoutingConfig()
) -> None:
    """Writes a `route_cost` attribute onto every edge of `graph`, in place,
    from an already-computed edge_potholes map (see `snap_potholes_to_edges`)."""
    for u, v, key, data in graph.edges(keys=True, data=True):
        # Falls back to raw length (mixing units) only if travel_time is
        # missing -- shouldn't happen for graphs from route_planner.graph,
        # but keeps this usable against ad-hoc/synthetic graphs.
        travel_time_s = data.get("travel_time", data.get("length", 0.0))
        exposure = sum(p.severity for p in edge_potholes.get((u, v, key), []))
        data[ROUTE_COST_ATTR] = travel_time_s + config.avoidance_weight * config.penalty_per_severity_s * exposure


def annotate_pothole_costs(
    graph, potholes: Iterable[PotholeReport], config: RoutingConfig = RoutingConfig()
) -> List[PotholeReport]:
    """Writes a `route_cost` attribute onto every edge of `graph`, in place.
    Returns the list of potholes that couldn't be matched to any nearby road."""
    edge_potholes, unmatched = snap_potholes_to_edges(graph, potholes, config)
    apply_edge_exposure_to_costs(graph, edge_potholes, config)
    return unmatched


def pothole_exposure_m(graph, u, v, potholes: Iterable[PotholeReport], config: RoutingConfig) -> float:
    """Total severity for a single edge (u, v)'s best-matching key, from the full snap assignment.
    Convenience wrapper for tests/inspection; prefer `annotate_pothole_costs` for routing."""
    edge_potholes, _ = snap_potholes_to_edges(graph, potholes, config)
    matches = [p for (eu, ev, _), matched in edge_potholes.items() if eu == u and ev == v for p in matched]
    return sum(p.severity for p in matches)
