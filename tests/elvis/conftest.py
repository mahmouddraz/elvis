from __future__ import annotations

import datetime as dt

import pytest

from core import TruncatedNormal, WeeklyProfileArrivals
from elvis import (
    ChargingDemand,
    ChargingEvent,
    EVBattery,
    InfrastructureSpec,
    Operation,
    Realisation,
    Scenario,
    TimeGrid,
    VehicleType,
)

MONDAY = dt.datetime(2024, 1, 1)


def make_vehicle(capacity_kwh: float = 50, max_power_kw: float = 11, **battery) -> VehicleType:
    return VehicleType(
        "Test", f"{capacity_kwh}kWh", EVBattery(capacity_kwh, max_power_kw, **battery), 1.0
    )


def make_scenario(
    infrastructure: InfrastructureSpec | None = None,
    *,
    operation: Operation | None = None,
    events_per_week: float = 100,
    **changes,
) -> Scenario:
    demand = ChargingDemand(
        WeeklyProfileArrivals(events_per_week, (1.0,) * 168, TruncatedNormal(4, 1, 0, 24)),
        TruncatedNormal(0.4, 0.1, 0, 1),
    )
    return Scenario(
        name="test",
        infrastructure=infrastructure or InfrastructureSpec.wallboxes(4, 11),
        vehicles=(make_vehicle(),),
        demand=demand,
        operation=operation or Operation(),
        **changes,
    )


def make_event(
    id: int,
    arrival_hours: float,
    parking_hours: float,
    soc: float = 0.5,
    vehicle: VehicleType | None = None,
) -> ChargingEvent:
    return ChargingEvent(
        id,
        MONDAY + dt.timedelta(hours=arrival_hours),
        parking_hours,
        soc,
        vehicle or make_vehicle(),
    )


def realisation_of(
    scenario: Scenario, events: list[ChargingEvent], hours: float = 24, resolution: str = "01:00:00"
) -> Realisation:
    grid = TimeGrid.create(MONDAY, MONDAY + dt.timedelta(hours=hours), resolution)
    return Realisation.from_events(scenario, grid, events)


@pytest.fixture
def week_grid() -> TimeGrid:
    return TimeGrid.create(
        MONDAY, MONDAY + dt.timedelta(days=7) - dt.timedelta(minutes=15), "15min"
    )
