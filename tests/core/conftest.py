"""Test helpers for ``core``, built on a deliberately non-EV toy domain: filling tanks."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from core import (
    BufferSpec,
    Demand,
    NetworkSpec,
    NodeSpec,
    Operation,
    Realisation,
    Request,
    ServerSpec,
    System,
    TimeGrid,
    TimeSeries,
    TruncatedNormal,
    WeeklyProfileArrivals,
)

MONDAY = dt.datetime(2024, 1, 1)


@dataclass(frozen=True)
class Tank:
    """A tank that holds ``capacity`` units and accepts at most ``max_rate`` units/hour."""

    name: str = "tank"
    capacity: float = 50.0
    max_rate: float = 11.0

    def max_rate_at(self, level: float) -> float:
        return self.max_rate


def star(
    servers: int = 4,
    server_rate: float = 11.0,
    *,
    root_rate: float | None = None,
    servers_per_node: int = 1,
    node_rate: float | None = None,
    buffer: BufferSpec | None = None,
) -> NetworkSpec:
    """One root with ``servers / servers_per_node`` inner nodes."""
    node = NodeSpec(
        node_rate or server_rate * servers_per_node, (ServerSpec(server_rate),) * servers_per_node
    )
    root = NodeSpec(
        root_rate or server_rate * servers, (node,) * (servers // servers_per_node), buffer=buffer
    )
    return NetworkSpec((root,))


def make_system(
    network: NetworkSpec | None = None,
    *,
    operation: Operation | None = None,
    arrivals_per_week: float = 100,
    background: TimeSeries | None = None,
) -> System:
    demand = Demand(
        WeeklyProfileArrivals(arrivals_per_week, (1.0,) * 168, TruncatedNormal(4, 1, 0, 24)),
        TruncatedNormal(0.4, 0.1, 0, 1),
        (Tank(),),
        (1.0,),
    )
    return System(network or star(), demand, operation or Operation(), background)


def make_request(
    id: int,
    arrival_hours: float,
    dwell_hours: float,
    level: float = 0.5,
    profile: Tank | None = None,
) -> Request:
    return Request(
        id, MONDAY + dt.timedelta(hours=arrival_hours), dwell_hours, level, profile or Tank()
    )


def realisation_of(
    system: System, requests: list[Request], hours: float = 24, resolution: str = "01:00:00"
) -> Realisation:
    grid = TimeGrid.create(MONDAY, MONDAY + dt.timedelta(hours=hours), resolution)
    return Realisation.from_requests(system, grid, requests)
