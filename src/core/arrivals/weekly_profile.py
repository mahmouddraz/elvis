"""Arrivals from a repeating weekly profile; dwell times from a truncated normal."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from core.config.validation import ProblemCollector
from core.random.samplers import Rng, TruncatedNormal
from core.time.grid import HOURS_PER_WEEK, TimeGrid, hour_of_week

SECONDS_PER_WEEK = HOURS_PER_WEEK * 3600

#: How the number of arrivals is drawn: ``fixed`` (the expected count, rounded up) or
#: ``poisson`` (a Poisson draw with the expected count as mean).
ARRIVAL_COUNTS = ("fixed", "poisson")


def check_arrival_count(problems: ProblemCollector, arrival_count: str) -> None:
    """Report an unknown ``arrival_count``."""
    if arrival_count not in ARRIVAL_COUNTS:
        problems.add(f"arrival_count must be one of {list(ARRIVAL_COUNTS)}, got {arrival_count!r}")


@dataclass(frozen=True)
class WeeklyProfileArrivals:
    """Arrival times follow a weekly profile; dwell times are independent.

    The profile holds relative weights for equally long slots covering Monday 00:00 to
    Sunday 24:00 (168 values for an hourly profile). Weights are linearly interpolated
    between slot starts and wrap around at the end of the week. Each time step receives an
    arrival probability proportional to its interpolated weight, and
    ``arrivals_per_week x weeks`` arrivals (rounded up) are drawn from those probabilities.
    With ``arrival_count="poisson"`` the number of arrivals is instead drawn from a Poisson
    distribution with mean ``arrivals_per_week x weeks``; given the count, arrival times are
    independent draws from the profile, which makes the arrivals an inhomogeneous Poisson
    process.

    Attributes:
        arrivals_per_week: Expected number of arrivals per week.
        weekly_profile: Relative arrival weights per slot of the week.
        dwell_hours: Time in the system, truncated to ``[0, max]``.
        arrival_count: ``"fixed"`` (default) or ``"poisson"``.
    """

    process: ClassVar[str] = "weekly_profile"

    arrivals_per_week: float
    weekly_profile: tuple[float, ...]
    dwell_hours: TruncatedNormal
    arrival_count: str = "fixed"

    def __post_init__(self) -> None:
        problems = ProblemCollector("weekly_profile arrivals")
        problems.non_negative("arrivals_per_week", self.arrivals_per_week)
        check_arrival_count(problems, self.arrival_count)
        if not self.weekly_profile or any(w < 0 for w in self.weekly_profile):
            problems.add("weekly_profile must be non-empty and non-negative")
        elif HOURS_PER_WEEK * 60 % len(self.weekly_profile):
            problems.add("weekly_profile length must split a week into whole-minute slots")
        problems.raise_if_any()

    def step_probabilities(self, grid: TimeGrid) -> np.ndarray:
        """Probability that a given arrival falls on each time step of ``grid``."""
        weights = np.asarray(self.weekly_profile, dtype=float)
        slot_hours = HOURS_PER_WEEK / len(weights)
        week_hours = (hour_of_week(grid.start) + np.arange(grid.num_steps) * grid.step_hours) % (
            HOURS_PER_WEEK
        )
        position = week_hours / slot_hours
        upper = np.ceil(position).astype(int)
        y0 = weights[(upper - 1) % len(weights)]
        y1 = weights[upper % len(weights)]
        step_weights = y0 + (y1 - y0) * (position - (upper - 1))
        total = step_weights.sum()
        if total == 0:
            return np.full(grid.num_steps, 1 / grid.num_steps)
        return step_weights / total

    def sample(self, grid: TimeGrid, rng: Rng) -> tuple[list[dt.datetime], np.ndarray]:
        """Arrival times (sorted, on grid steps) and dwell hours."""
        weeks = (grid.timestamps[-1] - grid.start).total_seconds() / SECONDS_PER_WEEK
        expected = self.arrivals_per_week * weeks
        count = (
            int(rng.poisson(expected)) if self.arrival_count == "poisson" else math.ceil(expected)
        )
        steps = np.sort(rng.choice(grid.num_steps, p=self.step_probabilities(grid), size=count))
        return [grid.timestamps[step] for step in steps], self.dwell_hours.sample(rng, count)

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> WeeklyProfileArrivals:
        keys = ("process", "arrivals_per_week", "weekly_profile", "dwell_hours")
        ProblemCollector.reject_unknown_keys(
            "weekly_profile arrivals", raw, (*keys, "arrival_count"), required=keys
        )
        dwell = raw["dwell_hours"]
        ProblemCollector.reject_unknown_keys(
            "dwell_hours", dwell, ("mean", "std", "max"), required=("mean", "std", "max")
        )
        return cls(
            arrivals_per_week=float(raw["arrivals_per_week"]),
            weekly_profile=tuple(float(w) for w in raw["weekly_profile"]),
            dwell_hours=TruncatedNormal(
                float(dwell["mean"]), float(dwell["std"]), 0.0, float(dwell["max"])
            ),
            arrival_count=raw.get("arrival_count", "fixed"),
        )

    def to_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "process": self.process,
            "arrivals_per_week": self.arrivals_per_week,
        }
        if self.arrival_count != "fixed":
            config["arrival_count"] = self.arrival_count
        config["dwell_hours"] = {**self.dwell_hours.to_config(), "max": self.dwell_hours.high}
        config["weekly_profile"] = list(self.weekly_profile)
        return config
