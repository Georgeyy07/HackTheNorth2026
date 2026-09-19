import networkx as nx

from route_planner.directions import build_directions


def build_l_shaped_graph():
    """A straight run north from A to B, then a hard turn east from B to C."""
    g = nx.MultiDiGraph()
    g.add_node("A", y=43.4600, x=-80.5300)
    g.add_node("B", y=43.4700, x=-80.5300)  # straight north of A
    g.add_node("C", y=43.4700, x=-80.5200)  # due east of B
    g.add_edge("A", "B", key=0, length=1000.0, name="Main Street")
    g.add_edge("B", "C", key=0, length=500.0, name="King Street")
    return g


def test_no_steps_for_a_single_node():
    assert build_directions(nx.MultiDiGraph(), ["A"]) == []


def test_straight_route_has_only_depart_and_arrive():
    g = nx.MultiDiGraph()
    g.add_node("A", y=43.4600, x=-80.5300)
    g.add_node("B", y=43.4700, x=-80.5300)
    g.add_node("C", y=43.4800, x=-80.5300)  # continues straight north
    g.add_edge("A", "B", key=0, length=500.0, name="Main Street")
    g.add_edge("B", "C", key=0, length=500.0, name="Main Street")

    steps = build_directions(g, ["A", "B", "C"])

    maneuvers = [s.maneuver for s in steps]
    assert maneuvers == ["depart", "arrive"]


def test_l_shaped_route_detects_the_turn_with_street_name():
    g = build_l_shaped_graph()

    steps = build_directions(g, ["A", "B", "C"])

    maneuvers = [s.maneuver for s in steps]
    assert maneuvers[0] == "depart"
    assert maneuvers[-1] == "arrive"
    turn_steps = [s for s in steps if s.maneuver not in ("depart", "arrive")]
    assert len(turn_steps) == 1
    turn = turn_steps[0]
    assert turn.maneuver in ("right", "sharp_right", "left", "sharp_left")
    assert turn.street == "King Street"
    assert turn.distance_m == 1000.0  # distance traveled on Main Street before the turn
    assert "King Street" in turn.instruction


def test_arrive_step_has_final_leg_distance():
    g = build_l_shaped_graph()

    steps = build_directions(g, ["A", "B", "C"])

    arrive = steps[-1]
    assert arrive.maneuver == "arrive"
    assert arrive.distance_m == 500.0
    assert arrive.lat == g.nodes["C"]["y"]
    assert arrive.lon == g.nodes["C"]["x"]
