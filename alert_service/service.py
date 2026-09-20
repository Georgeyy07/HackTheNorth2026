"""Runs the per-vehicle alert loop: GPS update -> nearby potholes -> compute_alert -> dispatch."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict

from .alert_math import AlertConfig, AlertResult, VehicleState, compute_alert
from .gps import GPSSource
from .potholes import PotholeStore

AlertSink = Callable[[str, AlertResult], Awaitable[None]]


class VehicleAlertLoop:
    """Consumes one vehicle's GPS stream and dispatches alerts as they trigger."""

    def __init__(
        self,
        vehicle_id: str,
        gps: GPSSource,
        potholes: PotholeStore,
        sink: AlertSink,
        config: AlertConfig = AlertConfig(),
    ) -> None:
        self.vehicle_id = vehicle_id
        self._gps = gps
        self._potholes = potholes
        self._sink = sink
        self._config = config

    async def run(self) -> None:
        async for vehicle in self._gps.updates():
            await self._tick(vehicle)

    async def _tick(self, vehicle: VehicleState) -> None:
        nearby = await self._potholes.query_nearby(
            vehicle.lat, vehicle.lon, self._config.max_alert_distance_m
        )
        for pothole in nearby:
            result = compute_alert(vehicle, pothole, self._config)
            if result.alert_level != "none":
                await self._sink(self.vehicle_id, result)


class AlertService:
    """Owns one VehicleAlertLoop per connected vehicle, running concurrently."""

    def __init__(
        self,
        potholes: PotholeStore,
        sink: AlertSink,
        config: AlertConfig = AlertConfig(),
    ) -> None:
        self._potholes = potholes
        self._sink = sink
        self._config = config
        self._tasks: Dict[str, asyncio.Task] = {}

    def add_vehicle(self, vehicle_id: str, gps: GPSSource) -> None:
        if vehicle_id in self._tasks:
            raise ValueError(f"vehicle {vehicle_id!r} already registered")
        loop = VehicleAlertLoop(vehicle_id, gps, self._potholes, self._sink, self._config)
        self._tasks[vehicle_id] = asyncio.create_task(loop.run())

    async def remove_vehicle(self, vehicle_id: str) -> None:
        task = self._tasks.pop(vehicle_id, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def wait_for(self, vehicle_id: str) -> None:
        """Awaits a single vehicle's loop task (it normally runs forever)."""
        await self._tasks[vehicle_id]
