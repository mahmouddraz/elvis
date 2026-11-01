"""Demand models: how requests arrive and what each one needs.

Any object with ``generate(grid, rng) -> list[Request]`` is a demand model
(:class:`DemandModel`). Two are built in:

* :class:`Demand` is parametric: an arrival process plus independently sampled initial
  levels, target levels and profiles.
* :class:`EmpiricalDemand` resamples measured sessions, so arrival time, dwell time, quantity
  and profile keep their joint distribution, including all correlations in the data.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from core.arrivals import ArrivalProcess
from core.errors import ConfigurationError
from core.random.samplers import CategoricalSampler, Rng, TruncatedNormal
from core.requests import Request, ServiceProfile
from core.time.grid import HOURS_PER_WEEK, TimeGrid, hour_of_week


class DemandModel(Protocol):
    """Anything that generates requests for a time grid."""

    def generate(self, grid: TimeGrid, rng: Rng) -> list[Request]: ...


@dataclass(frozen=True)
class Demand:
    """Generates requests: arrivals and dwell times, then initial levels, then profiles.

    The attributes are sampled independently, in that order, from one generator; target
    levels are sampled last and only if a distribution is given.

    Attributes:
        arrivals: When requests arrive and how long they may stay.
        initial_level: Fill level on arrival, truncated to [0, 1].
        profiles: Kinds of request (e.g. vehicle models).
        weights: Relative frequency of each profile; normalised internally.
        target_level: Level at which service stops; ``None`` means 1 for every request.
    """

    arrivals: ArrivalProcess
    initial_level: TruncatedNormal
    profiles: tuple[ServiceProfile, ...]
    weights: tuple[float, ...]
    target_level: TruncatedNormal | None = None

    def __post_init__(self) -> None:
        if not self.profiles or len(self.profiles) != len(self.weights):
            raise ConfigurationError("demand: need one weight per profile, and >= 1 profile")
        names = [profile.name for profile in self.profiles]
        if len(set(names)) != len(names):
            raise ConfigurationError("demand: profile names must be unique")
        CategoricalSampler(self.weights)  # validates the weights

    def generate(self, grid: TimeGrid, rng: Rng) -> list[Request]:
        arrivals, dwell = self.arrivals.sample(grid, rng)
        levels = self.initial_level.sample(rng, len(arrivals))
        profiles = self.sample_profiles(rng, len(arrivals))
        targets = (
            np.ones(len(arrivals))
            if self.target_level is None
            else self.target_level.sample(rng, len(arrivals))
        )
        return [
            Request(
                i, arrival, float(dwell_hours), float(min(level, target)), profile, float(target)
            )
            for i, (arrival, dwell_hours, level, profile, target) in enumerate(
                zip(arrivals, dwell, levels, profiles, targets, strict=True), start=1
            )
        ]

    def sample_profiles(self, rng: Rng, size: int) -> Sequence[ServiceProfile]:
        indices = CategoricalSampler(self.weights).sample(rng, size)
        return [self.profiles[i] for i in indices]


@dataclass(frozen=True)
class EmpiricalDemand:
    """Resamples measured sessions with replacement (bootstrap).

    Each session is ``(hour of week of arrival, dwell hours, quantity, profile index)``.
    A resampled session keeps all four together; it is placed in a random week of the
    simulated period at its hour of week and starts at the next time step. The initial level
    is ``1 - quantity / capacity`` of its profile (clipped at 0), the target level 1, so the
    request needs exactly the measured quantity.

    Attributes:
        arrivals_per_week: Expected number of sessions per week.
        sessions: Array of shape ``(n, 4)`` as described above.
        profiles: The profiles referenced by index.
    """

    arrivals_per_week: float
    sessions: np.ndarray
    profiles: tuple[ServiceProfile, ...]

    def __post_init__(self) -> None:
        sessions = np.asarray(self.sessions, dtype=float)
        if sessions.ndim != 2 or sessions.shape[1] != 4 or len(sessions) == 0:
            raise ConfigurationError("empirical demand: sessions must be an (n, 4) array")
        if np.any(sessions[:, 0] < 0) or np.any(sessions[:, 0] >= HOURS_PER_WEEK):
            raise ConfigurationError("empirical demand: hour of week must lie in [0, 168)")
        if np.any(sessions[:, 1:3] < 0):
            raise ConfigurationError("empirical demand: dwell and quantity must be >= 0")
        indices = sessions[:, 3]
        if (
            np.any(indices != np.round(indices))
            or indices.min() < 0
            or indices.max() >= len(self.profiles)
        ):
            raise ConfigurationError("empirical demand: profile index out of range")
        for profile in self.profiles:
            if not (np.isfinite(profile.capacity) and profile.capacity > 0):
                raise ConfigurationError(f"empirical demand: profile {profile.name!r} capacity")
            if not (np.isfinite(profile.max_rate_at(0.0)) and profile.max_rate_at(0.0) > 0):
                raise ConfigurationError(f"empirical demand: profile {profile.name!r} rate")
        object.__setattr__(self, "sessions", sessions)

    def generate(self, grid: TimeGrid, rng: Rng) -> list[Request]:
        start_hours = hour_of_week(grid.start)
        week_start = grid.start - dt.timedelta(hours=start_hours)
        weeks = math.ceil((start_hours + grid.duration_hours) / HOURS_PER_WEEK) + 1
        draws = rng.poisson(self.arrivals_per_week * weeks)
        picked = self.sessions[rng.integers(len(self.sessions), size=draws)]
        week = rng.integers(weeks, size=draws)
        offsets = week * HOURS_PER_WEEK + picked[:, 0]
        keep = (offsets >= start_hours) & (offsets <= start_hours + grid.duration_hours)
        order = np.argsort(offsets[keep], kind="stable")
        picked, offsets = picked[keep][order], offsets[keep][order]
        steps_per_hour = 1 / grid.step_hours
        offsets = np.ceil(offsets * steps_per_hour - 1e-9) / steps_per_hour
        requests = []
        for i, (offset, (_, dwell, quantity, index)) in enumerate(
            zip(offsets, picked, strict=True), start=1
        ):
            profile = self.profiles[int(index)]
            level = max(0.0, 1 - quantity / profile.capacity)
            requests.append(
                Request(
                    i, week_start + dt.timedelta(hours=float(offset)), float(dwell), level, profile
                )
            )
        return requests
