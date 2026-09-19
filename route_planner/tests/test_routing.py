import networkx as nx

from alert_service.alert_math import PotholeReport
from route_planner.cost import RoutingConfig, pothole_exposure, pothole_exposure_m
from route_planner.router import find_route


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


def test_zero_avoidance_weight_matches_pure_shortest_path():
    graph = build_diamond_graph()
    pothole_at_b = PotholeReport(id="p1", lat=43.4700, lon=-80.5430, severity=1.0)

    routes = find_route(graph, "A", "D", [pothole_at_b], RoutingConfig(avoidance_weight=0.0))

    assert routes["efficient"].nodes == ["A", "B", "D"]
    assert routes["pothole_aware"].nodes == ["A", "B", "D"]


def test_strong_avoidance_takes_the_longer_pothole_free_route():
    graph = build_diamond_graph()
    pothole_at_b = PotholeReport(id="p1", lat=43.4700, lon=-80.5430, severity=1.0)
    config = RoutingConfig(avoidance_weight=5.0, penalty_per_severity_s=1000.0)

    routes = find_route(graph, "A", "D", [pothole_at_b], config)

    assert routes["efficient"].nodes == ["A", "B", "D"]       # baseline ignores the pothole
    assert routes["pothole_aware"].nodes == ["A", "C", "D"]   # detours around it
    assert routes["pothole_aware"].distance_m > routes["efficient"].distance_m
    assert routes["pothole_aware"].pothole_exposure < routes["efficient"].pothole_exposure


def test_mild_avoidance_still_prefers_the_short_route_for_a_minor_pothole():
    graph = build_diamond_graph()
    minor_pothole = PotholeReport(id="p1", lat=43.4700, lon=-80.5430, severity=0.1)
    config = RoutingConfig(avoidance_weight=0.5, penalty_per_severity_s=500.0)

    routes = find_route(graph, "A", "D", [minor_pothole], config)

    # penalty here (0.5*500*0.1=25s) is smaller than the 30s extra the detour
    # costs (70s vs 40s), so a low-severity pothole isn't worth going around
    assert routes["pothole_aware"].nodes == ["A", "B", "D"]


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
