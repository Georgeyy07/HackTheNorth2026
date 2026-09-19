import pytest

from alert_service.alert_math import (
    AlertConfig,
    PotholeReport,
    VehicleState,
    angular_delta_deg,
    bearing_deg,
    compute_alert,
    eta_seconds,
    haversine_distance_m,
)


def test_haversine_zero_for_identical_points():
    assert haversine_distance_m(43.4723, -80.5449, 43.4723, -80.5449) == 0.0


def test_haversine_matches_known_degree_of_latitude():
    # One degree of latitude is ~111.32km almost everywhere on Earth,
    # independent of the haversine formula itself -- a standard geodesy fact.
    d = haversine_distance_m(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(111_320, rel=0.01)


def test_haversine_matches_known_city_distance():
    # Waterloo, ON to downtown Toronto, ON is a published ~94km straight-line
    # distance -- an independent real-world check, not derived from this code.
    waterloo = (43.4643, -80.5204)
    toronto = (43.6532, -79.3832)
    d = haversine_distance_m(*waterloo, *toronto)
    assert d == pytest.approx(94_000, rel=0.03)


@pytest.mark.parametrize(
    "target_lat,target_lon,expected_bearing",
    [
        (1.0, 0.0, 0.0),      # due north
        (0.0, 1.0, 90.0),     # due east
        (-1.0, 0.0, 180.0),   # due south
        (0.0, -1.0, 270.0),   # due west
    ],
)
def test_bearing_matches_cardinal_directions(target_lat, target_lon, expected_bearing):
    bearing = bearing_deg(0.0, 0.0, target_lat, target_lon)
    assert bearing == pytest.approx(expected_bearing, abs=0.5)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (0.0, 0.0, 0.0),
        (0.0, 180.0, 180.0),
        (350.0, 10.0, 20.0),   # wraps around 0/360
        (10.0, 350.0, 20.0),   # symmetric
        (90.0, 270.0, 180.0),
    ],
)
def test_angular_delta_handles_wraparound(a, b, expected):
    assert angular_delta_deg(a, b) == pytest.approx(expected)


def test_eta_seconds_none_when_stopped():
    assert eta_seconds(distance_m=100.0, speed_mps=0.0) is None


def test_eta_seconds_divides_distance_by_speed():
    assert eta_seconds(distance_m=100.0, speed_mps=10.0) == pytest.approx(10.0)


def test_compute_alert_warns_when_heading_straight_at_a_close_pothole():
    # Same geometry as alert_math.py's own __main__ sanity check.
    vehicle = VehicleState(lat=43.4723, lon=-80.5449, heading_deg=90.0, speed_mps=13.4)
    pothole = PotholeReport(id="p1", lat=43.4723, lon=-80.5430)

    result = compute_alert(vehicle, pothole)

    assert result.is_approaching is True
    assert result.distance_m == pytest.approx(153.3, abs=1.0)
    assert result.eta_seconds == pytest.approx(11.4, abs=0.5)
    assert result.alert_level == "watch"  # ETA > warn_eta_seconds (8s default)


def test_compute_alert_warns_when_eta_is_short():
    vehicle = VehicleState(lat=43.4723, lon=-80.5449, heading_deg=90.0, speed_mps=13.4)
    pothole = PotholeReport(id="p1", lat=43.4723, lon=-80.5440)  # closer: ~80m

    result = compute_alert(vehicle, pothole)

    assert result.alert_level == "warn"
    assert result.eta_seconds <= AlertConfig().warn_eta_seconds


def test_compute_alert_ignores_pothole_behind_the_vehicle():
    vehicle = VehicleState(lat=43.4723, lon=-80.5449, heading_deg=90.0, speed_mps=13.4)
    pothole = PotholeReport(id="behind", lat=43.4723, lon=-80.5470)  # west, vehicle faces east

    result = compute_alert(vehicle, pothole)

    assert result.is_approaching is False
    assert result.alert_level == "none"


def test_compute_alert_ignores_stopped_vehicle():
    vehicle = VehicleState(lat=43.4723, lon=-80.5449, heading_deg=90.0, speed_mps=0.0)
    pothole = PotholeReport(id="p1", lat=43.4723, lon=-80.5440)

    result = compute_alert(vehicle, pothole)

    assert result.is_approaching is False
    assert result.eta_seconds is None
    assert result.alert_level == "none"


def test_compute_alert_ignores_pothole_beyond_max_distance():
    vehicle = VehicleState(lat=43.4723, lon=-80.5449, heading_deg=90.0, speed_mps=13.4)
    pothole = PotholeReport(id="far", lat=43.4723, lon=-80.4000)  # several km east

    result = compute_alert(vehicle, pothole, AlertConfig(max_alert_distance_m=400.0))

    assert result.is_approaching is False
    assert result.alert_level == "none"
