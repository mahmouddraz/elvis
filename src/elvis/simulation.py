"""Simulating charging scenarios."""

from __future__ import annotations

import datetime as dt

import numpy as np

from core.engine import ProgressCallback, Simulation
from core.errors import ConfigurationError
from core.time.grid import TimeGrid
from elvis.realisation import Realisation
from elvis.result import ChargingResult
from elvis.scenario import Scenario


def simulate(
    scenario: Scenario | Realisation,
    start: str | dt.datetime | None = None,
    end: str | dt.datetime | None = None,
    resolution: str | dt.timedelta = "00:15:00",
    *,
    seed: int | np.random.Generator | None = None,
    progress: ProgressCallback | None = None,
) -> ChargingResult:
    """Simulate a scenario over a period, or re-simulate an existing realisation.

    Args:
        scenario: A :class:`Scenario` (events are sampled first) or a :class:`Realisation`
            (its events and period are used as they are).
        start: First time step, e.g. ``"2024-01-01"``; required for a scenario.
        end: Last time step; required for a scenario.
        resolution: Step length, e.g. ``"00:15:00"`` or ``"15min"``.
        seed: Seed or generator for event sampling. Use a seed for reproducible results.
        progress: Called with the completed fraction in [0, 1] during the run.

    Example:
        >>> from elvis import load_preset, simulate
        >>> result = simulate(load_preset("office"), "2024-01-01", "2024-01-07 23:45", seed=1)
        >>> result.peak_load_kw > 0
        True
    """
    if isinstance(scenario, Realisation):
        if start is not None or end is not None:
            raise ConfigurationError("a realisation already fixes the period; omit start/end")
        realisation = scenario
    else:
        if start is None or end is None:
            raise ConfigurationError("start and end are required to simulate a scenario")
        realisation = scenario.realise(TimeGrid.create(start, end, resolution), seed)
    return ChargingResult(realisation, Simulation(realisation.core).run(progress))
