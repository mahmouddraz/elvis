"""Running simulations."""

from __future__ import annotations

import datetime as dt

import numpy as np

from core.engine.engine import ProgressCallback, Simulation
from core.engine.queue import WaitingQueue
from core.engine.result import Result, SessionRecord
from core.errors import ConfigurationError
from core.system import Realisation, System
from core.time.grid import TimeGrid


def simulate(
    system: System | Realisation,
    start: str | dt.datetime | None = None,
    end: str | dt.datetime | None = None,
    resolution: str | dt.timedelta = "00:15:00",
    *,
    seed: int | np.random.Generator | None = None,
    progress: ProgressCallback | None = None,
) -> Result:
    """Simulate a system over a period, or re-simulate an existing realisation.

    Args:
        system: A :class:`System` (requests are sampled first) or a :class:`Realisation`
            (its requests and period are used as they are).
        start: First time step; required for a system.
        end: Last time step; required for a system.
        resolution: Step length, e.g. ``"00:15:00"`` or ``"15min"``.
        seed: Seed or generator for sampling. Use a seed for reproducible results.
        progress: Called with the completed fraction in [0, 1] during the run.
    """
    if isinstance(system, Realisation):
        if start is not None or end is not None:
            raise ConfigurationError("a realisation already fixes the period; omit start/end")
        realisation = system
    else:
        if start is None or end is None:
            raise ConfigurationError("start and end are required to simulate a system")
        realisation = system.realise(TimeGrid.create(start, end, resolution), seed)
    return Simulation(realisation).run(progress)


__all__ = [
    "ProgressCallback",
    "Result",
    "SessionRecord",
    "Simulation",
    "WaitingQueue",
    "simulate",
]
