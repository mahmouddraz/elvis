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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from core.arrivals import ArrivalProcess
from core.errors import ConfigurationError
from core.random.samplers import CategoricalSampler, Rng, TruncatedNormal
from core.requests import LEVEL_TOLERANCE_DECIMALS, Request, ServiceProfile
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

    A profile index may be NaN (profile not measured); such sessions draw their profile from
    ``weights``, independently of the other session attributes.

    Attributes:
        arrivals_per_week: Expected number of sessions per week.
        sessions: Array of shape ``(n, 4)`` as described above.
        profiles: The profiles referenced by index.
        weights: Relative profile frequencies for sessions without a profile index.
        supply_side: If True, quantities are measured on the supply side (e.g. at the meter)
            and are multiplied by the profile's ``efficiency`` (default 1) to obtain the
            quantity the request needs.
        feasible: Optional test ``feasible(profile, quantity, dwell_hours, step_hours)``
            (quantity on the request side). If given, a session without a profile index
            draws its profile from ``weights`` restricted to the feasible profiles; if none is
            feasible it draws from all ``weights`` and is counted as infeasible.
    """

    arrivals_per_week: float
    sessions: np.ndarray
    profiles: tuple[ServiceProfile, ...]
    weights: tuple[float, ...] | None = None
    supply_side: bool = False
    feasible: Callable[[ServiceProfile, float, float, float], bool] | None = field(
        default=None, compare=False
    )

    def __post_init__(self) -> None:
        sessions = np.asarray(self.sessions, dtype=float)
        if sessions.ndim != 2 or sessions.shape[1] != 4 or len(sessions) == 0:
            raise ConfigurationError("empirical demand: sessions must be an (n, 4) array")
        if not np.isfinite(sessions[:, :3]).all():
            raise ConfigurationError("empirical demand: sessions must be finite numbers")
        if np.any(sessions[:, 0] < 0) or np.any(sessions[:, 0] >= HOURS_PER_WEEK):
            raise ConfigurationError("empirical demand: hour of week must lie in [0, 168)")
        if np.any(sessions[:, 1:3] < 0):
            raise ConfigurationError("empirical demand: dwell and quantity must be >= 0")
        indices = sessions[:, 3]
        if np.isnan(indices).any():
            if self.weights is None or len(self.weights) != len(self.profiles):
                raise ConfigurationError(
                    "empirical demand: sessions without profile need one weight per profile"
                )
            CategoricalSampler(self.weights)
            indices = indices[~np.isnan(indices)]
        if len(indices) and (
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
        return self.generate_with_info(grid, rng)[0]

    def generate_with_info(self, grid: TimeGrid, rng: Rng) -> tuple[list[Request], dict[str, Any]]:
        """Requests plus counts of how profiles were drawn for sessions without one.

        Keys: ``sessions_without_profile``, ``drawn_conditionally`` (feasibility-restricted
        draws) and ``infeasible`` (no profile feasible; drawn from all weights).
        """
        week_start, offsets, picked = place_weekly(self.sessions, self.arrivals_per_week, grid, rng)
        missing = np.isnan(picked[:, 3])
        info = {
            "sessions_without_profile": int(missing.sum()),
            "drawn_conditionally": 0,
            "infeasible": 0,
        }
        if missing.any():
            assert self.weights is not None
            if self.feasible is None:
                picked[missing, 3] = CategoricalSampler(self.weights).sample(
                    rng, int(missing.sum())
                )
            else:
                self._draw_feasible(picked, missing, grid.step_hours, rng, info)
        requests = []
        for i, (offset, (_, dwell, quantity, index)) in enumerate(
            zip(offsets, picked, strict=True), start=1
        ):
            profile = self.profiles[int(index)]
            level = max(0.0, 1 - self._needed(quantity, profile) / profile.capacity)
            requests.append(
                Request(
                    i, week_start + dt.timedelta(hours=float(offset)), float(dwell), level, profile
                )
            )
        return requests, info

    def _draw_feasible(
        self,
        picked: np.ndarray,
        missing: np.ndarray,
        step_hours: float,
        rng: Rng,
        info: dict[str, Any],
    ) -> None:
        """Draw the profile of each session in ``missing`` among the feasible profiles."""
        assert self.weights is not None
        assert self.feasible is not None
        weights = np.asarray(self.weights, dtype=float)
        for row in np.flatnonzero(missing):
            _, dwell, quantity, _ = picked[row]
            mask = np.array(
                [
                    self.feasible(profile, self._needed(quantity, profile), dwell, step_hours)
                    for profile in self.profiles
                ]
            )
            restricted = weights * mask
            if restricted.sum() > 0:
                info["drawn_conditionally"] += 1
                p = restricted / restricted.sum()
            else:
                info["infeasible"] += 1
                p = weights / weights.sum()
            picked[row, 3] = rng.choice(len(self.profiles), p=p)

    def _needed(self, quantity: float, profile: ServiceProfile) -> float:
        """Request-side quantity of a measured session for ``profile``."""
        if self.supply_side:
            return quantity * float(getattr(profile, "efficiency", 1.0))
        return quantity


def place_weekly(
    rows: np.ndarray, per_week: float, grid: TimeGrid, rng: Rng
) -> tuple[dt.datetime, np.ndarray, np.ndarray]:
    """Resample rows whose first column is an hour of the week onto ``grid``.

    A Poisson number of rows (mean ``per_week`` per week) is drawn with replacement; each is
    placed in a random week at its hour of week and moved to the next time step. Returns the
    start of the first week, the sorted offsets in hours from it, and the drawn rows.
    """
    start_hours = hour_of_week(grid.start)
    week_start = grid.start - dt.timedelta(hours=start_hours)
    weeks = math.ceil((start_hours + grid.duration_hours) / HOURS_PER_WEEK) + 1
    draws = rng.poisson(per_week * weeks)
    picked = rows[rng.integers(len(rows), size=draws)]
    offsets = rng.integers(weeks, size=draws) * HOURS_PER_WEEK + picked[:, 0]
    keep = (offsets >= start_hours) & (offsets <= start_hours + grid.duration_hours)
    order = np.argsort(offsets[keep], kind="stable")
    picked, offsets = picked[keep][order], offsets[keep][order]
    steps_per_hour = 1 / grid.step_hours
    return week_start, np.ceil(offsets * steps_per_hour - 1e-9) / steps_per_hour, picked


def reaches_target(
    profile: ServiceProfile,
    quantity: float,
    dwell_hours: float,
    step_hours: float,
    server_rate: float,
    kind: str | None = None,
) -> bool:
    """Whether an uncongested server delivers ``quantity`` (request side) within the dwell.

    Replays the engine's physics for one request alone on a server with ``server_rate`` and
    ``kind``: the same rate curve and efficiency as :class:`core.network.runtime.Session`,
    the rate held constant within a step, ``ceil(dwell / step)`` steps (at least one), and
    the engine's completion tolerance. The request needs ``quantity <= capacity``.
    """
    capacity = profile.capacity
    if quantity > capacity:
        return False
    efficiency = float(getattr(profile, "efficiency", 1.0))
    rate_curve = getattr(profile, "rate_curve", None)
    curve = profile.max_rate_at if kind is None or rate_curve is None else rate_curve(kind)
    level = 1 - quantity / capacity
    steps = max(1, math.ceil(dwell_hours / step_hours - 1e-9))
    for _ in range(steps):
        if round(level, LEVEL_TOLERANCE_DECIMALS) >= 1:
            return True
        rate = min(
            server_rate,
            curve(level) / efficiency,
            (1 - level) * capacity / step_hours / efficiency,
        )
        level = min(1, level + rate * efficiency * step_hours / capacity)
    return round(level, LEVEL_TOLERANCE_DECIMALS) >= 1
