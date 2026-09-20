"""GPS input sources for the alert service.

A GPSSource yields VehicleState updates as they arrive. QueueGPSSource is the
"real" implementation for now: phones POST location pings to the HTTP API
(see api.py), which pushes them onto a per-vehicle queue that the alert loop
consumes. SimulatedGPSSource replays a scripted path, for local development
without a phone in hand.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Iterable

from .alert_math import VehicleState


class GPSSource:
    """Yields VehicleState updates for one vehicle, in arrival order."""

    async def updates(self) -> AsyncIterator[VehicleState]:
        raise NotImplementedError
        yield  # pragma: no cover - makes this an async generator signature for subclasses


class QueueGPSSource(GPSSource):
    """Backed by an asyncio.Queue that an API handler pushes onto."""

    def __init__(self) -> None:
        self._queue: "asyncio.Queue[VehicleState]" = asyncio.Queue()

    def push(self, state: VehicleState) -> None:
        self._queue.put_nowait(state)

    async def updates(self) -> AsyncIterator[VehicleState]:
        while True:
            yield await self._queue.get()


class SimulatedGPSSource(GPSSource):
    """Replays a fixed sequence of states, one per `interval_s`. For tests/demos."""

    def __init__(self, states: Iterable[VehicleState], interval_s: float = 1.0) -> None:
        self._states = list(states)
        self._interval_s = interval_s

    async def updates(self) -> AsyncIterator[VehicleState]:
        for state in self._states:
            yield state
            if self._interval_s:
                await asyncio.sleep(self._interval_s)
