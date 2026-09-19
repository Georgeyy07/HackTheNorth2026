"""CLI demo: type two addresses, get a map with multiple selectable route
options (Fastest / Balanced / Avoid potholes), each rated by the actual
potholes it passes, using real potholes from TigerDB.

Usage:
    python -m route_planner.demo "200 University Ave W, Waterloo, ON" "King St & Erb St, Waterloo, ON" \\
        --place "Waterloo, Ontario, Canada" --output route_planner/demo_output/route.html
"""

from __future__ import annotations

import argparse
import asyncio

import folium

from alert_service.potholes import TigerDBPotholeStore
from route_planner.cost import RoutingConfig
from route_planner.graph import geocode_address, load_road_graph, nearest_node
from route_planner.router import find_routes

ROUTE_COLORS = ["crimson", "seagreen", "royalblue", "darkorange"]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origin", help="Starting address")
    parser.add_argument("destination", help="Destination address")
    parser.add_argument("--place", default="Waterloo, Ontario, Canada", help="City/region to load the road network for")
    parser.add_argument("--pothole-radius-m", type=float, default=5000.0, help="How far around the origin to pull potholes from TigerDB")
    parser.add_argument("--output", default="route_planner/demo_output/route.html", help="Where to save the map")
    args = parser.parse_args()

    print(f"Geocoding '{args.origin}' and '{args.destination}'...")
    origin_lat, origin_lon = geocode_address(args.origin)
    dest_lat, dest_lon = geocode_address(args.destination)
    print(f"  origin:      {origin_lat:.5f}, {origin_lon:.5f}")
    print(f"  destination: {dest_lat:.5f}, {dest_lon:.5f}")

    print(f"Loading road network for '{args.place}'...")
    graph = load_road_graph(args.place)

    print("Fetching nearby potholes from TigerDB...")
    store = TigerDBPotholeStore()
    potholes = await store.query_nearby(origin_lat, origin_lon, radius_m=args.pothole_radius_m)
    print(f"  {len(potholes)} potholes in range")

    origin_node = nearest_node(graph, origin_lat, origin_lon)
    dest_node = nearest_node(graph, dest_lat, dest_lon)

    result = find_routes(graph, origin_node, dest_node, potholes, config=RoutingConfig())
    routes = result["routes"]

    for r in routes:
        pothole_ids = ", ".join(p.id for p in r.potholes_encountered) or "none"
        print(f"{r.label:>16}: {r.distance_m:.0f}m, {r.duration_s:.0f}s ETA, risk={r.risk_rating}, potholes=[{pothole_ids}]")
    if result["unmatched_potholes"]:
        print(f"  ({len(result['unmatched_potholes'])} pothole reports too far from any road to route around)")

    m = folium.Map(location=routes[0].coords[len(routes[0].coords) // 2], zoom_start=14)
    for r, color in zip(routes, ROUTE_COLORS):
        folium.PolyLine(
            r.coords, color=color, weight=5, opacity=0.85,
            tooltip=f"{r.label}: {r.distance_m:.0f}m, {r.duration_s:.0f}s, risk={r.risk_rating}",
        ).add_to(m)
    folium.Marker(routes[0].coords[0], tooltip=args.origin, icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker(routes[0].coords[-1], tooltip=args.destination, icon=folium.Icon(color="blue")).add_to(m)
    for p in potholes:
        folium.CircleMarker(
            [p.lat, p.lon], radius=6, color="black", fill=True, fill_color="red",
            tooltip=f"Pothole {p.id} (severity {p.severity})",
        ).add_to(m)

    m.save(args.output)
    print(f"Saved map to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
