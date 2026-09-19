"""
Cross-device alert math for the pothole detection app.

Given a vehicle's current position/heading/speed and a known pothole's
position, this module answers three questions:
  1. How far away is the pothole? (great-circle distance, meters)
  2. Is the vehicle actually heading toward it? (bearing comparison)
  3. If so, how long until it gets there? (speed-based ETA)

These combine into a single alert decision so the live map / notification
layer can just call `compute_alert(...)` and get back everything it needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


EARTH_RADIUS_M = 6_371_000.0


@dataclass
class VehicleState:
    lat: float
    lon: float
    heading_deg: float   # compass bearing the vehicle is traveling, 0-360, 0 = North
    speed_mps: float      # ground speed in meters/second (0 if stopped)


@dataclass
class PotholeReport:
    id: str
    lat: float
    lon: float
    severity: float = 1.0  # 0-1, used later by the risk-score layer


@dataclass
class AlertConfig:
    max_alert_distance_m: float = 400.0   # ignore potholes farther than this
    heading_tolerance_deg: float = 35.0    # how far off-heading still counts as "approaching"
    warn_eta_seconds: float = 8.0          # ETA below this triggers an alert
    min_speed_mps: float = 0.5             # below this, treat as stationary (no ETA-based alert)


@dataclass
class AlertResult:
    pothole_id: str
    distance_m: float
    bearing_to_pothole_deg: float
    heading_delta_deg: float
    is_approaching: bool
    eta_seconds: Optional[float]     # None if not approaching or stationary
    should_alert: bool
    alert_level: str                 # "none" | "watch" | "warn"


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing (0-360, 0 = North) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)

    x = math.sin(d_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)

    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360) % 360


def angular_delta_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two compass bearings (0-180)."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def eta_seconds(distance_m: float, speed_mps: float) -> Optional[float]:
    """Straight-line ETA. Returns None if speed is ~0 (can't estimate)."""
    if speed_mps <= 0:
        return None
    return distance_m / speed_mps


def compute_alert(
    vehicle: VehicleState,
    pothole: PotholeReport,
    config: AlertConfig = AlertConfig(),
) -> AlertResult:
    """Decide whether `vehicle` should be alerted about `pothole`, and how urgently."""

    distance = haversine_distance_m(vehicle.lat, vehicle.lon, pothole.lat, pothole.lon)
    bearing_to_pothole = bearing_deg(vehicle.lat, vehicle.lon, pothole.lat, pothole.lon)
    heading_delta = angular_delta_deg(vehicle.heading_deg, bearing_to_pothole)

    is_approaching = (
        distance <= config.max_alert_distance_m
        and heading_delta <= config.heading_tolerance_deg
        and vehicle.speed_mps >= config.min_speed_mps
    )

    eta = eta_seconds(distance, vehicle.speed_mps) if is_approaching else None

    should_alert = is_approaching and eta is not None and eta <= config.warn_eta_seconds

    if should_alert:
        alert_level = "warn"
    elif is_approaching:
        alert_level = "watch"
    else:
        alert_level = "none"

    return AlertResult(
        pothole_id=pothole.id,
        distance_m=round(distance, 1),
        bearing_to_pothole_deg=round(bearing_to_pothole, 1),
        heading_delta_deg=round(heading_delta, 1),
        is_approaching=is_approaching,
        eta_seconds=round(eta, 1) if eta is not None else None,
        should_alert=should_alert,
        alert_level=alert_level,
    )


if __name__ == "__main__":
    # Quick sanity check: vehicle heading roughly toward a pothole 150m ahead.
    vehicle = VehicleState(lat=43.4723, lon=-80.5449, heading_deg=90.0, speed_mps=13.4)  # ~48 km/h
    pothole = PotholeReport(id="pothole-001", lat=43.4723, lon=-80.5430, severity=0.7)

    result = compute_alert(vehicle, pothole)
    print(result)
