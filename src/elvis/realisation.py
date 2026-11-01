"""Realisation: a scenario with one concrete sample of charging events for a time grid.

Simulating the same realisation under different policies isolates the effect of the policy
from sampling noise. Realisations can be saved and reloaded to share exact inputs.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.requests import Request
from core.system import Realisation as CoreRealisation
from core.time.durations import format_duration
from core.time.grid import TimeGrid
from elvis.scenario import Scenario


@dataclass(frozen=True)
class Realisation:
    """A scenario plus its sampled (or measured) charging events.

    Attributes:
        scenario: The scenario the events belong to.
        core: The equivalent :class:`core.Realisation`, whose requests are the events.
    """

    scenario: Scenario
    core: CoreRealisation

    @property
    def grid(self) -> TimeGrid:
        return self.core.grid

    @property
    def events(self) -> tuple[Request, ...]:
        """Charging events (core requests) sorted by arrival."""
        return self.core.requests

    @property
    def seed(self) -> int | None:
        return self.core.seed

    @classmethod
    def sample(
        cls, scenario: Scenario, grid: TimeGrid, seed: int | np.random.Generator | None = None
    ) -> Realisation:
        """Draw events from the scenario's demand; ``seed`` may be a generator."""
        return cls(scenario, scenario.system.realise(grid, seed))

    @classmethod
    def from_events(cls, scenario: Scenario, grid: TimeGrid, events: list[Request]) -> Realisation:
        """Use measured or hand-made events instead of the scenario's demand model."""
        return cls(scenario, CoreRealisation.from_requests(scenario.system, grid, events))

    def with_scenario(self, scenario: Scenario) -> Realisation:
        """The same events under a different scenario, e.g. another scheduling policy."""
        return Realisation(
            scenario, CoreRealisation(scenario.system, self.grid, self.events, self.seed)
        )

    def preload_kw(self) -> np.ndarray:
        """Transformer preload per time step (zeros if the scenario defines none)."""
        return self.core.background()

    def emission_factors(self) -> np.ndarray | None:
        """Grid emission factor per time step in g/kWh, if the scenario defines one."""
        series = self.scenario.emission_factors
        return None if series is None else series.align(self.grid)

    def to_record(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.to_config(),
            "grid": {
                "start": self.grid.start.isoformat(),
                "end": self.grid.end.isoformat(),
                "resolution": format_duration(self.grid.resolution),
            },
            "seed": self.seed,
            "events": [event.to_record() for event in self.events],
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Realisation:
        scenario = Scenario.from_config(record["scenario"])
        grid = TimeGrid.create(**record["grid"])
        vehicles = {vehicle.name: vehicle for vehicle in scenario.vehicles}
        events = tuple(Request.from_record(e, vehicles) for e in record["events"])
        return cls(scenario, CoreRealisation(scenario.system, grid, events, record.get("seed")))

    def save(self, path: str | Path) -> None:
        """Write JSON; a ``.gz`` suffix compresses it."""
        path = Path(path)
        data = json.dumps(self.to_record()).encode("utf-8")
        path.write_bytes(gzip.compress(data) if path.suffix == ".gz" else data)

    @classmethod
    def load(cls, path: str | Path) -> Realisation:
        """Read a file written by :meth:`save`."""
        path = Path(path)
        data = path.read_bytes()
        if path.suffix == ".gz":
            data = gzip.decompress(data)
        return cls.from_record(json.loads(data))
