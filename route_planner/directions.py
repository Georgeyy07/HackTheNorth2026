"""Turn-by-turn directions from a route's node sequence.

A turn instruction is just "how much does the road's bearing change at this
intersection" -- the exact same geometry alert_math already uses to decide
if a car is heading toward a pothole (bearing_deg between two points). This
reuses that rather than inventing separate turn-detection math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from alert_service.alert_math import bearing_deg

STRAIGHT_THRESHOLD_DEG = 20.0
SLIGHT_THRESHOLD_DEG = 45.0
SHARP_THRESHOLD_DEG = 120.0


@dataclass
class DirectionStep:
    instruction: str
    maneuver: str  # "depart" | "straight" | "slight_left" | "left" | "sharp_left" |
                   # "slight_right" | "right" | "sharp_right" | "arrive"
    street: Optional[str]
    distance_m: float  # distance traveled since the previous step, up to this one
    lat: float
    lon: float


def _signed_bearing_delta(from_bearing: float, to_bearing: float) -> float:
    """Signed turn angle in (-180, 180]. Positive = right turn, negative = left turn."""
    return (to_bearing - from_bearing + 180) % 360 - 180


def _classify_turn(delta_signed: float) -> tuple[str, str]:
    abs_delta = abs(delta_signed)
    if abs_delta <= STRAIGHT_THRESHOLD_DEG:
        return "straight", "Continue straight"
    side = "right" if delta_signed > 0 else "left"
    if abs_delta >= SHARP_THRESHOLD_DEG:
        return f"sharp_{side}", f"Make a sharp {side}"
    if abs_delta <= SLIGHT_THRESHOLD_DEG:
        return f"slight_{side}", f"Turn slight {side}"
    return side, f"Turn {side}"


def _edge_data(graph, u, v):
    return min(graph.get_edge_data(u, v).values(), key=lambda d: d.get("length", 0.0))


def _edge_length(graph, u, v) -> float:
    return _edge_data(graph, u, v).get("length", 0.0)


def _edge_name(graph, u, v) -> Optional[str]:
    name = _edge_data(graph, u, v).get("name")
    if isinstance(name, list):
        return name[0] if name else None
    return name


def build_directions(graph, nodes: List) -> List[DirectionStep]:
    """Walks a route's node sequence and emits one step per real turn (plus
    depart/arrive), each carrying the distance since the previous step, the
    maneuver point's coordinates, and the street being turned onto."""
    if len(nodes) < 2:
        return []

    first_lat, first_lon = graph.nodes[nodes[0]]["y"], graph.nodes[nodes[0]]["x"]
    steps = [
        DirectionStep(
            instruction=f"Head out on {_edge_name(graph, nodes[0], nodes[1]) or 'the road'}",
            maneuver="depart",
            street=_edge_name(graph, nodes[0], nodes[1]),
            distance_m=0.0,
            lat=first_lat, lon=first_lon,
        )
    ]

    accumulated = 0.0
    for i in range(1, len(nodes) - 1):
        a, b, c = nodes[i - 1], nodes[i], nodes[i + 1]
        alat, alon = graph.nodes[a]["y"], graph.nodes[a]["x"]
        blat, blon = graph.nodes[b]["y"], graph.nodes[b]["x"]
        clat, clon = graph.nodes[c]["y"], graph.nodes[c]["x"]

        incoming_bearing = bearing_deg(alat, alon, blat, blon)
        outgoing_bearing = bearing_deg(blat, blon, clat, clon)
        delta = _signed_bearing_delta(incoming_bearing, outgoing_bearing)
        accumulated += _edge_length(graph, a, b)

        maneuver, phrase = _classify_turn(delta)
        if maneuver != "straight":
            street = _edge_name(graph, b, c)
            instruction = phrase + (f" onto {street}" if street else "")
            steps.append(DirectionStep(
                instruction=instruction, maneuver=maneuver, street=street,
                distance_m=accumulated, lat=blat, lon=blon,
            ))
            accumulated = 0.0

    accumulated += _edge_length(graph, nodes[-2], nodes[-1])
    last_lat, last_lon = graph.nodes[nodes[-1]]["y"], graph.nodes[nodes[-1]]["x"]
    steps.append(DirectionStep(
        instruction="Arrive at your destination", maneuver="arrive", street=None,
        distance_m=accumulated, lat=last_lat, lon=last_lon,
    ))
    return steps
