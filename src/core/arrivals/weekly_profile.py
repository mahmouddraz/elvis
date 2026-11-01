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


@dataclass(frozen=True)
class WeeklyProfileArrivals:
    """Arrival times follow a weekly profile; dwell times are independent.

    The profile holds relative weights for equally long slots covering Monday 00:00 to
    Sunday 24:00 (168 values for an hourly profile). Weights are linearly interpolated
    between slot starts and wrap around at the end of the week. Each time step receives an
    arrival probability proportional to its interpolated weight, and
    ``arrivals_per_week x weeks`` arrivals are drawn from those probabilities.

    Attributes:
        arrivals_per_week: Expected number of arrivals per week.
        weekly_profile: Relative arrival weights per slot of the week.
        dwell_hours: Time in the system, truncated to ``[0, max]``.
    """

    process: ClassVar[str] = "weekly_profile"

    arrivals_per_week: float
    weekly_profile: tuple[float, ...]
    dwell_hours: TruncatedNormal

    def __post_init__(self) -> None:
        problems = ProblemCollector("weekly_profile arrivals")
        problems.non_negative("arrivals_per_week", self.arrivals_per_week)
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
        count = math.ceil(self.arrivals_per_week * weeks)
        steps = np.sort(rng.choice(grid.num_steps, p=self.step_probabilities(grid), size=count))
        return [grid.timestamps[step] for step in steps], self.dwell_hours.sample(rng, count)

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> WeeklyProfileArrivals:
        keys = ("process", "arrivals_per_week", "weekly_profile", "dwell_hours")
        ProblemCollector.reject_unknown_keys("weekly_profile arrivals", raw, keys, required=keys)
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
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "process": self.process,
            "arrivals_per_week": self.arrivals_per_week,
            "dwell_hours": {**self.dwell_hours.to_config(), "max": self.dwell_hours.high},
            "weekly_profile": list(self.weekly_profile),
        }
