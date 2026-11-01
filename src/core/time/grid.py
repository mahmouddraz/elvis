"""The discrete time axis of a simulation."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from core.errors import ConfigurationError
from core.time.durations import parse_datetime, parse_duration

HOURS_PER_DAY = 24
HOURS_PER_WEEK = 7 * HOURS_PER_DAY
SECONDS_PER_HOUR = 3600


def hour_of_week(timestamp: dt.datetime) -> float:
    """Hours elapsed since Monday 00:00 of the week containing ``timestamp``."""
    return (
        timestamp.weekday() * HOURS_PER_DAY
        + timestamp.hour
        + timestamp.minute / 60
        + timestamp.second / 60 / 60
    )


def hour_of_day(timestamp: dt.datetime) -> float:
    """Hours elapsed since midnight."""
    return timestamp.hour + timestamp.minute / 60 + timestamp.second / 60 / 60


@dataclass(frozen=True)
class TimeGrid:
    """Equally spaced time steps from ``start`` to ``end`` (inclusive, if on the grid)."""

    start: dt.datetime
    end: dt.datetime
    resolution: dt.timedelta

    def __post_init__(self) -> None:
        if self.resolution <= dt.timedelta(0):
            raise ConfigurationError(f"resolution must be positive, got {self.resolution}")
        if self.end < self.start:
            raise ConfigurationError(f"end ({self.end}) lies before start ({self.start})")

    @classmethod
    def create(
        cls,
        start: str | dt.datetime,
        end: str | dt.datetime,
        resolution: str | dt.timedelta,
    ) -> TimeGrid:
        """Build a grid from strings or datetime objects."""
        return cls(
            parse_datetime(start, name="start"),
            parse_datetime(end, name="end"),
            parse_duration(resolution, name="resolution"),
        )

    @cached_property
    def num_steps(self) -> int:
        """Number of time steps."""
        return int((self.end - self.start) / self.resolution) + 1

    @cached_property
    def step_hours(self) -> float:
        """Length of one time step in hours."""
        return self.resolution.total_seconds() / SECONDS_PER_HOUR

    @cached_property
    def duration_hours(self) -> float:
        """Hours between the first and the last time step."""
        return (self.timestamps[-1] - self.start).total_seconds() / SECONDS_PER_HOUR

    @cached_property
    def timestamps(self) -> tuple[dt.datetime, ...]:
        """All time steps as datetime objects."""
        return tuple(self.start + i * self.resolution for i in range(self.num_steps))

    @cached_property
    def datetime64(self) -> np.ndarray:
        """All time steps as ``datetime64[us]``, e.g. for building data frames."""
        return np.array(self.timestamps, dtype="datetime64[us]")

    def step_of(self, timestamp: dt.datetime) -> int:
        """Index of the first time step at or after ``timestamp``."""
        return int(np.ceil((timestamp - self.start) / self.resolution))
