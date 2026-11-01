"""Arrival processes: when requests arrive and how long they may stay.

An arrival process is any object with

* ``sample(grid, rng) -> (arrival datetimes sorted ascending, dwell hours array)``,
* ``to_config() -> dict`` containing a ``"process"`` key, and
* a ``from_config(raw)`` classmethod.

Register new processes with :func:`register_arrival_process` to use them in input files.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Protocol

import numpy as np

from core.arrivals.gaussian_mixture import GaussianMixtureArrivals, MixtureComponent
from core.arrivals.weekly_profile import WeeklyProfileArrivals
from core.errors import ConfigurationError
from core.random.samplers import Rng
from core.time.grid import TimeGrid


class ArrivalProcess(Protocol):
    """Interface every arrival process implements."""

    process: str

    def sample(self, grid: TimeGrid, rng: Rng) -> tuple[list[dt.datetime], np.ndarray]: ...

    def to_config(self) -> dict[str, Any]: ...


ARRIVAL_PROCESSES: dict[str, Any] = {
    WeeklyProfileArrivals.process: WeeklyProfileArrivals,
    GaussianMixtureArrivals.process: GaussianMixtureArrivals,
}


def register_arrival_process(cls: Any) -> Any:
    """Make an arrival process class usable from input files; usable as a decorator."""
    ARRIVAL_PROCESSES[cls.process] = cls
    return cls


def arrival_process_from_config(raw: dict[str, Any]) -> ArrivalProcess:
    """Build the arrival process named by ``raw["process"]``."""
    name = raw.get("process") if isinstance(raw, dict) else None
    if name not in ARRIVAL_PROCESSES:
        raise ConfigurationError(
            f"arrivals: process must be one of {sorted(ARRIVAL_PROCESSES)}, got {name!r}"
        )
    process: ArrivalProcess = ARRIVAL_PROCESSES[name].from_config(raw)
    return process


__all__ = [
    "ARRIVAL_PROCESSES",
    "ArrivalProcess",
    "GaussianMixtureArrivals",
    "MixtureComponent",
    "WeeklyProfileArrivals",
    "arrival_process_from_config",
    "register_arrival_process",
]
