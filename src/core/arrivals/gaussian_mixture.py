"""Arrivals and dwell times jointly from a Gaussian mixture over (hour of week, dwell hours).

Unlike :class:`~core.arrivals.weekly_profile.WeeklyProfileArrivals`, the mixture captures
the dependence between arrival time and dwell time, e.g. that morning arrivals stay longer.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from core.config.validation import ProblemCollector
from core.errors import ConfigurationError
from core.random.samplers import Rng
from core.time.grid import HOURS_PER_DAY, HOURS_PER_WEEK, TimeGrid, hour_of_week

REPLACEMENT_POOL_SIZE = 100


@dataclass(frozen=True, slots=True)
class MixtureComponent:
    """One bivariate normal component.

    Attributes:
        weight: Mixing weight; all weights of a mixture sum to 1.
        mean: ``(arrival hour of week, dwell hours)``.
        covariance: 2x2 covariance matrix.
    """

    weight: float
    mean: tuple[float, float]
    covariance: tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class GaussianMixtureArrivals:
    """Arrival time and dwell time from a 2-D Gaussian mixture fitted on weekly data.

    Attributes:
        arrivals_per_week: Number of arrivals sampled per week.
        components: The mixture components.
        min_dwell_hours: Draws with shorter dwell times are replaced by fresh draws.
        max_dwell_hours: Longer dwell times are clipped.
        day_shift_hours: The mixture was fitted on arrival times shifted back by this many
            hours within each day (so a "day" can start at e.g. 19:00 and overnight arrivals
            are not split at midnight). Draws are shifted forward again, wrapping within
            their day.
    """

    process: ClassVar[str] = "gaussian_mixture"

    arrivals_per_week: int
    components: tuple[MixtureComponent, ...]
    min_dwell_hours: float = 0.167
    max_dwell_hours: float = 24.0
    day_shift_hours: float = 5.0

    def __post_init__(self) -> None:
        problems = ProblemCollector("gaussian_mixture arrivals")
        problems.non_negative("arrivals_per_week", self.arrivals_per_week)
        weights = [component.weight for component in self.components]
        if not self.components or not 0.99 < sum(weights) < 1.01:
            problems.add("component weights must sum to 1 (1 % tolerance)")
        if any(not 0 < component.mean[0] < HOURS_PER_WEEK for component in self.components):
            problems.add("component arrival means must lie within the week (0, 168) hours")
        problems.raise_if_any()

    def sample(self, grid: TimeGrid, rng: Rng) -> tuple[list[dt.datetime], np.ndarray]:
        """Arrival times (sorted, rounded up to grid steps) and dwell hours."""
        draws = self._sample_weeks(grid, rng)
        start_hours = hour_of_week(grid.start)
        in_period = (draws[:, 0] >= start_hours) & (
            draws[:, 0] <= start_hours + grid.duration_hours
        )
        draws = draws[in_period]
        week_start = grid.start - dt.timedelta(hours=start_hours)
        arrivals = [week_start + dt.timedelta(hours=hours) for hours in draws[:, 0]]
        return arrivals, np.minimum(draws[:, 1], self.max_dwell_hours)

    def _sample_weeks(self, grid: TimeGrid, rng: Rng) -> np.ndarray:
        """``(hours since the first Monday 00:00, dwell hours)`` for whole weeks, sorted."""
        pool = _ReplacementPool(self, rng)
        steps_per_hour = 1 / grid.step_hours
        weeks = []
        for week in range(_weeks_to_sample(grid)):
            draws = self._draw(rng, self.arrivals_per_week)
            for i in np.flatnonzero(draws[:, 1] < self.min_dwell_hours):
                draws[i] = pool.take()
            draws[:, 0] += week * HOURS_PER_WEEK
            draws[:, 0] = np.ceil(draws[:, 0] * steps_per_hour) / steps_per_hour
            weeks.append(draws)
        draws = np.concatenate(weeks)
        return draws[np.lexsort((draws[:, 1], draws[:, 0]))]

    def _draw(self, rng: Rng, size: int) -> np.ndarray:
        """Draw from the mixture and undo the day shift."""
        counts = rng.multinomial(size, [c.weight for c in self.components])
        draws = np.vstack(
            [
                rng.multivariate_normal(component.mean, component.covariance, int(count))
                for component, count in zip(self.components, counts, strict=True)
            ]
        )
        draws[:, 0] = _unshift_within_day(draws[:, 0], self.day_shift_hours)
        return draws

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> GaussianMixtureArrivals:
        optional = ("min_dwell_hours", "max_dwell_hours", "day_shift_hours")
        required = ("process", "arrivals_per_week", "components")
        ProblemCollector.reject_unknown_keys(
            "gaussian_mixture arrivals", raw, required + optional, required=required
        )
        components = []
        for entry in raw["components"]:
            ProblemCollector.reject_unknown_keys(
                "mixture component",
                entry,
                ("weight", "mean", "covariance"),
                required=("weight", "mean", "covariance"),
            )
            covariance = np.asarray(entry["covariance"], dtype=float)
            if np.shape(entry["mean"]) != (2,) or covariance.shape != (2, 2):
                raise ConfigurationError("mixture component: mean must be 2-D, covariance 2x2")
            c = covariance.tolist()
            components.append(
                MixtureComponent(
                    float(entry["weight"]),
                    (float(entry["mean"][0]), float(entry["mean"][1])),
                    ((c[0][0], c[0][1]), (c[1][0], c[1][1])),
                )
            )
        return cls(
            arrivals_per_week=int(raw["arrivals_per_week"]),
            components=tuple(components),
            **{key: float(raw[key]) for key in optional if key in raw},
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "process": self.process,
            "arrivals_per_week": self.arrivals_per_week,
            "min_dwell_hours": self.min_dwell_hours,
            "max_dwell_hours": self.max_dwell_hours,
            "day_shift_hours": self.day_shift_hours,
            "components": [
                {
                    "weight": c.weight,
                    "mean": list(c.mean),
                    "covariance": [list(row) for row in c.covariance],
                }
                for c in self.components
            ],
        }


class _ReplacementPool:
    """Valid draws used to replace draws whose dwell time is too short.

    Refilled in batches, so rejection sampling costs one vectorised draw per batch.
    """

    def __init__(self, mixture: GaussianMixtureArrivals, rng: Rng) -> None:
        self._mixture = mixture
        self._rng = rng
        self._refill()

    def _refill(self) -> None:
        draws = self._mixture._draw(self._rng, REPLACEMENT_POOL_SIZE)
        self._draws = draws[draws[:, 1] >= self._mixture.min_dwell_hours]
        self._next = 0

    def take(self) -> np.ndarray:
        draw = self._draws[self._next].copy()
        self._next += 1
        if self._next == len(self._draws):
            self._refill()
        return draw


def _unshift_within_day(hours: np.ndarray, shift: float) -> np.ndarray:
    """Add ``shift`` hours, wrapping back to the start of the same day if it overflows.

    Values at or beyond the end of the week are left untouched.
    """
    day_end = np.maximum((np.floor(hours / HOURS_PER_DAY) + 1) * HOURS_PER_DAY, HOURS_PER_DAY)
    wraps = hours > day_end - shift
    shifted = np.where(wraps, hours - (HOURS_PER_DAY - shift), hours + shift)
    return np.where(hours < HOURS_PER_WEEK, shifted, hours)


def _weeks_to_sample(grid: TimeGrid) -> int:
    """Calendar weeks touched by the simulation period."""
    first, last = grid.start, grid.timestamps[-1]
    weeks = math.ceil(grid.duration_hours / HOURS_PER_WEEK)
    if first.weekday() > last.weekday() or (
        first.weekday() == last.weekday() and first.time() > last.time()
    ):
        weeks += 1
    return weeks
