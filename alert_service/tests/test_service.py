import asyncio

from alert_service.alert_math import AlertConfig, AlertResult, PotholeReport, VehicleState
from alert_service.gps import SimulatedGPSSource
from alert_service.potholes import InMemoryPotholeStore
from alert_service.service import AlertService


def run(coro):
    return asyncio.run(coro)


def test_loop_dispatches_alerts_for_states_that_trigger_them():
    pothole = PotholeReport(id="p1", lat=43.4723, lon=-80.5430, severity=0.7)
    store = InMemoryPotholeStore([pothole])

    approaching = VehicleState(lat=43.4723, lon=-80.5449, heading_deg=90.0, speed_mps=13.4)
    stopped = VehicleState(lat=43.4723, lon=-80.5449, heading_deg=90.0, speed_mps=0.0)
    gps = SimulatedGPSSource([approaching, stopped], interval_s=0)

    dispatched: list[tuple[str, AlertResult]] = []

    async def sink(vehicle_id: str, result: AlertResult) -> None:
        dispatched.append((vehicle_id, result))

    async def scenario():
        service = AlertService(store, sink, AlertConfig())
        service.add_vehicle("car-1", gps)
        await service.wait_for("car-1")

    run(scenario())

    assert len(dispatched) == 1
    vehicle_id, result = dispatched[0]
    assert vehicle_id == "car-1"
    assert result.pothole_id == "p1"
    assert result.alert_level in ("watch", "warn")


def test_loop_ignores_potholes_outside_max_alert_distance():
    far_pothole = PotholeReport(id="far", lat=43.60, lon=-80.10)
    store = InMemoryPotholeStore([far_pothole])
    gps = SimulatedGPSSource(
        [VehicleState(lat=43.4723, lon=-80.5449, heading_deg=90.0, speed_mps=13.4)],
        interval_s=0,
    )

    dispatched = []

    async def sink(vehicle_id, result):
        dispatched.append(result)

    async def scenario():
        service = AlertService(store, sink)
        service.add_vehicle("car-1", gps)
        await service.wait_for("car-1")

    run(scenario())

    assert dispatched == []


def test_in_memory_store_filters_by_radius():
    near = PotholeReport(id="near", lat=43.4723, lon=-80.5430)
    far = PotholeReport(id="far", lat=43.60, lon=-80.10)
    store = InMemoryPotholeStore([near, far])

    results = run(store.query_nearby(43.4723, -80.5449, radius_m=400.0))

    assert [p.id for p in results] == ["near"]
