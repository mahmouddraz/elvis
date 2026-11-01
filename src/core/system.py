"""System and realisation: what is simulated, and one concrete sample of its demand."""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from core.allocation import Policy, policy_from_config
from core.demand import DemandModel
from core.errors import ConfigurationError
from core.network.spec import NetworkSpec
from core.random.samplers import make_rng
from core.requests import Request
from core.time.grid import TimeGrid, hour_of_day
from core.time.series import TimeSeries

ReleaseRule = Literal["at_departure", "at_target"]


@dataclass(frozen=True, slots=True)
class OpenHours:
    """Daily window, in hours since midnight, during which the system is open (inclusive)."""

    opens: float
    closes: float

    def __post_init__(self) -> None:
        if not 0 <= self.opens <= self.closes <= 24:
            raise ConfigurationError("open hours: need 0 <= opens <= closes <= 24")

    def contains(self, timestamp: dt.datetime) -> bool:
        return self.opens <= hour_of_day(timestamp) <= self.closes


@dataclass(frozen=True)
class Operation:
    """Operating rules.

    Attributes:
        policy: Policy name or ``{policy: name, **parameters}``; see :mod:`core.allocation`.
        queue_capacity: Requests that may wait for a free server (0: no queue). Waiting
            requests leave when their dwell time is over.
        release: ``"at_departure"``: requests occupy their server until their dwell time is
            over. ``"at_target"``: they leave as soon as they reach their target level,
            freeing the server for waiting requests.
        open_hours: Outside these hours arrivals are rejected and all requests leave.
    """

    policy: str | Mapping[str, Any] = "unconstrained"
    queue_capacity: int = 0
    release: ReleaseRule = "at_departure"
    open_hours: OpenHours | None = None

    def __post_init__(self) -> None:
        if self.queue_capacity < 0:
            raise ConfigurationError("operation: queue_capacity must be >= 0")
        if self.release not in ("at_departure", "at_target"):
            raise ConfigurationError("operation: release must be 'at_departure' or 'at_target'")
        self.create_policy()  # fail early on an unknown policy or bad parameters

    def create_policy(self) -> Policy:
        """A fresh policy instance (policies may be stateful within a run)."""
        return policy_from_config(self.policy)


@dataclass(frozen=True)
class System:
    """Everything that defines what is simulated, independent of the simulated period.

    Attributes:
        network: Servers and the capacity limits above them.
        demand: How requests arrive and what they need.
        operation: Policy, queueing and opening hours.
        background: Load on each root node that is not caused by requests.
    """

    network: NetworkSpec
    demand: DemandModel
    operation: Operation = field(default_factory=Operation)
    background: TimeSeries | None = None

    def replace(self, **changes: Any) -> System:
        """A copy with some attributes changed."""
        return dataclasses.replace(self, **changes)

    def realise(self, grid: TimeGrid, seed: int | np.random.Generator | None = None) -> Realisation:
        """Sample requests for ``grid``; ``seed`` may be an int or a generator."""
        requests = tuple(self.demand.generate(grid, make_rng(seed)))
        return Realisation(self, grid, requests, seed if isinstance(seed, int) else None)


@dataclass(frozen=True)
class Realisation:
    """A system with one concrete set of requests for a time grid.

    Simulating the same realisation under different policies isolates the effect of the
    policy from sampling noise.

    Attributes:
        system: The simulated system.
        grid: The simulated time steps.
        requests: Requests sorted by arrival.
        seed: Seed the requests were sampled with, if known.
    """

    system: System
    grid: TimeGrid
    requests: tuple[Request, ...]
    seed: int | None = None

    def __post_init__(self) -> None:
        arrivals = [request.arrival for request in self.requests]
        if arrivals != sorted(arrivals):
            raise ConfigurationError("realisation: requests must be sorted by arrival")

    @classmethod
    def from_requests(cls, system: System, grid: TimeGrid, requests: list[Request]) -> Realisation:
        """Use measured or hand-made requests instead of the system's demand model."""
        return cls(system, grid, tuple(sorted(requests, key=lambda request: request.arrival)))

    def background(self) -> np.ndarray:
        """Background load per time step (zeros if none is defined)."""
        series = self.system.background
        return np.zeros(self.grid.num_steps) if series is None else series.align(self.grid)
