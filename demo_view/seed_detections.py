"""CLI & Utility script to seed or reset simulated detections for the pitch demo."""
import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from road_viewer.tiger_db import (
    init_db,
    seed_simulated_detections,
    clear_simulated_detections,
    get_simulated_detections,
    get_potholes,
)


def main():
    parser = argparse.ArgumentParser(description="Seed mock fleet trajectories into simulated_detections table.")
    parser.add_argument("--clear-only", action="store_true", help="Only clear existing simulated detections without seeding")
    parser.add_argument("--append", action="store_true", help="Do not clear existing detections before seeding")
    args = parser.parse_args()

    init_db()

    if args.clear_only:
        count = clear_simulated_detections()
        print(f"Cleared {count} simulated detections.")
        return

    records = seed_simulated_detections(clear_existing=not args.append)
    print(f"Successfully seeded {len(records)} simulated detections across 3 vehicles in Waterloo!")
    
    # Summary of vehicles
    cars = set(r["car_id"] for r in records)
    anomalies = [r for r in records if r["imu"] or r["yolo"]]
    dual = [r for r in records if r["imu"] and r["yolo"]]
    
    print(f"Fleet vehicles: {len(cars)} ({', '.join(cars)})")
    print(f"Total waypoints: {len(records)}")
    print(f"Normal GPS tracking points: {len(records) - len(anomalies)}")
    print(f"Total anomalies: {len(anomalies)} (Dual confirmed: {len(dual)})")
    print(f"Active potholes currently in database: {len(get_potholes())}")


if __name__ == "__main__":
    main()
