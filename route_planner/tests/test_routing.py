import networkx as nx

from alert_service.alert_math import PotholeReport
from route_planner.cost import RoutingConfig, pothole_exposure, pothole_exposure_m, snap_potholes_to_edges
from route_planner.router import find_routes


def build_diamond_graph():
    """A -> B -> D is the short route (400m). A -> C -> D is a 700m detour.
    A pothole sits right at B, on the short route."""
    g = nx.MultiDiGraph()
    coords = {
        "A": (43.4700, -80.5450),
        "B": (43.4700, -80.5430),
        "C": (43.4680, -80.5440),
        "D": (43.4700, -80.5410),
    }
    for node, (lat, lon) in coords.items():
        g.add_node(node, y=lat, x=lon)

    # Uniform 10 m/s (~36 km/h) assumed speed, so travel_time is just length/10.
    g.add_edge("A", "B", key=0, length=200.0, travel_time=20.0)
    g.add_edge("B", "D", key=0, length=200.0, travel_time=20.0)
    g.add_edge("A", "C", key=0, length=350.0, travel_time=35.0)
    g.add_edge("C", "D", key=0, length=350.0, travel_time=35.0)
    return g


def routes_by_label(result):
    return {r.label: r for r in result["routes"]}


def test_zero_avoidance_weight_matches_pure_shortest_path():
    graph = build_diamond_graph()
    pothole_at_b = PotholeReport(id="p1", lat=43.4700, lon=-80.5430, severity=1.0)

    result = find_routes(
        graph, "A", "D", [pothole_at_b],
        presets=[("efficient", 0.0), ("pothole_aware", 0.0)],
    )
    routes = routes_by_label(result)

    # identical presets collapse to a single route, since it's not a real
    # alternative for the user to pick
    assert len(result["routes"]) == 1
    assert routes["efficient"].nodes == ["A", "B", "D"]


def test_strong_avoidance_takes_the_longer_pothole_free_route():
    graph = build_diamond_graph()
    pothole_at_b = PotholeReport(id="p1", lat=43.4700, lon=-80.5430, severity=1.0)
    config = RoutingConfig(penalty_per_severity_s=1000.0)

    result = find_routes(
        graph, "A", "D", [pothole_at_b],
        presets=[("efficient", 0.0), ("pothole_aware", 5.0)],
        config=config,
    )
    routes = routes_by_label(result)

    assert routes["efficient"].nodes == ["A", "B", "D"]       # baseline ignores the pothole
    assert routes["pothole_aware"].nodes == ["A", "C", "D"]   # detours around it
    assert routes["pothole_aware"].distance_m > routes["efficient"].distance_m
    assert routes["pothole_aware"].pothole_exposure < routes["efficient"].pothole_exposure
    assert routes["efficient"].risk_rating == "CRITICAL"      # drives straight through severity-1.0
    assert routes["pothole_aware"].risk_rating == "None"
    assert [p.id for p in routes["efficient"].potholes_encountered] == ["p1"]
    assert routes["pothole_aware"].potholes_encountered == []


def test_mild_avoidance_still_prefers_the_short_route_for_a_minor_pothole():
    graph = build_diamond_graph()
    minor_pothole = PotholeReport(id="p1", lat=43.4700, lon=-80.5430, severity=0.1)
    config = RoutingConfig(penalty_per_severity_s=500.0)

    result = find_routes(graph, "A", "D", [minor_pothole], presets=[("pothole_aware", 0.5)], config=config)

    # penalty here (0.5*500*0.1=25s) is smaller than the 30s extra the detour
    # costs (70s vs 40s), so a low-severity pothole isn't worth going around
    assert result["routes"][0].nodes == ["A", "B", "D"]
    assert result["routes"][0].risk_rating == "LOW"


def test_find_routes_returns_multiple_distinct_options_for_the_user_to_pick():
    graph = build_diamond_graph()
    pothole_at_b = PotholeReport(id="p1", lat=43.4700, lon=-80.5430, severity=1.0)
    # Tuned so "Balanced" (weight 2) stays on the short route (2*10=20s added,
    # short route still 60s < 70s detour) but "Avoid potholes" (weight 8)
    # crosses the threshold (8*10=80s added, 120s > 70s detour) -- otherwise
    # they'd collapse to the same route and there'd be nothing to pick between.
    config = RoutingConfig(penalty_per_severity_s=10.0)

    result = find_routes(graph, "A", "D", [pothole_at_b], config=config)  # default 3 presets

    labels = [r.label for r in result["routes"]]
    assert "Fastest" in labels
    assert "Avoid potholes" in labels
    # Fastest and Avoid potholes must actually differ, or there's nothing to pick between
    node_sequences = {tuple(r.nodes) for r in result["routes"]}
    assert len(node_sequences) == len(result["routes"])


def test_unmatched_potholes_are_reported_separately():
    graph = build_diamond_graph()
    far_pothole = PotholeReport(id="far", lat=43.60, lon=-80.10, severity=1.0)

    result = find_routes(graph, "A", "D", [far_pothole], config=RoutingConfig(max_snap_distance_m=100.0))

    assert [p.id for p in result["unmatched_potholes"]] == ["far"]
    assert result["routes"][0].potholes_encountered == []


def test_pothole_exposure_ignores_potholes_outside_the_buffer():
    far_pothole = PotholeReport(id="far", lat=43.60, lon=-80.10, severity=1.0)
    exposure = pothole_exposure(43.4700, -80.5430, [far_pothole], buffer_m=30.0)
    assert exposure == 0.0


def test_pothole_exposure_sums_multiple_nearby_potholes():
    at_point = [
        PotholeReport(id="a", lat=43.4700, lon=-80.5430, severity=0.5),
        PotholeReport(id="b", lat=43.4700, lon=-80.5430, severity=0.25),
    ]
    exposure = pothole_exposure(43.4700, -80.5430, at_point, buffer_m=30.0)
    assert exposure == 0.75


def test_edge_exposure_catches_a_pothole_mid_segment_far_from_both_endpoints():
    # A long, mostly-east-west edge with a pothole sitting near its midpoint --
    # ~275m from each endpoint, well outside a 30m endpoint-only buffer, but
    # only ~a few meters off the segment's actual line.
    graph = nx.MultiDiGraph()
    graph.add_node("A", y=43.4700, x=-80.5500)
    graph.add_node("B", y=43.4700, x=-80.5400)
    graph.add_edge("A", "B", key=0, length=800.0)

    midpoint_pothole = PotholeReport(id="mid", lat=43.4700, lon=-80.5450, severity=1.0)

    exposure = pothole_exposure_m(graph, "A", "B", [midpoint_pothole], RoutingConfig(max_snap_distance_m=30.0))

    assert exposure == 1.0


def test_pothole_applies_to_both_directions_of_a_two_way_street():
    # Real road graphs store a two-way street as two separate directed edges
    # with near-identical geometry. A pothole must affect routing whichever
    # direction the road is traveled, not just whichever direction happened
    # to be checked first (this was a real bug: confirmed via the live
    # Waterloo graph, a pothole snapped only to A->B and was invisible to a
    # route traveling B->A down the same physical street).
    graph = nx.MultiDiGraph()
    graph.add_node("A", y=43.4700, x=-80.5500)
    graph.add_node("B", y=43.4700, x=-80.5400)
    graph.add_edge("A", "B", key=0, length=800.0)
    graph.add_edge("B", "A", key=0, length=800.0)

    pothole = PotholeReport(id="mid", lat=43.4700, lon=-80.5450, severity=1.0)

    edge_potholes, unmatched = snap_potholes_to_edges(graph, [pothole], RoutingConfig(max_snap_distance_m=30.0))

    assert unmatched == []
    assert [p.id for p in edge_potholes.get(("A", "B", 0), [])] == ["mid"]
    assert [p.id for p in edge_potholes.get(("B", "A", 0), [])] == ["mid"]
