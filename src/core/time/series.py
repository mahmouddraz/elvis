"""Exogenous time series aligned to a time grid (e.g. background load, prices, intensities)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.errors import ConfigurationError
from core.time.durations import format_duration, parse_duration
from core.time.grid import TimeGrid


@dataclass(frozen=True)
class TimeSeries:
    """A series of values that starts at the beginning of the simulation.

    Attributes:
        values: One value per ``resolution`` step, or a single constant.
        resolution: Spacing of ``values``. ``None`` means "same as the simulation".
        repeat: Repeat ``values`` periodically if they are shorter than the simulation.
    """

    values: tuple[float, ...]
    resolution: dt.timedelta | None = None
    repeat: bool = False

    def __post_init__(self) -> None:
        if not self.values:
            raise ConfigurationError("time series must contain at least one value")
        if not np.all(np.isfinite(self.values)):
            raise ConfigurationError("time series contains non-finite values")

    @classmethod
    def constant(cls, value: float) -> TimeSeries:
        """A series that has the same value at every time step."""
        return cls((float(value),), repeat=True)

    @classmethod
    def from_config(cls, raw: Any, *, name: str) -> TimeSeries | None:
        """Parse a scalar, a list, or ``{values, resolution, repeat}``; ``None`` passes through."""
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return cls.constant(raw)
        if isinstance(raw, Sequence) and not isinstance(raw, str):
            return cls(_as_floats(raw, name))
        if isinstance(raw, dict):
            unknown = set(raw) - {"values", "resolution", "repeat"}
            if unknown:
                raise ConfigurationError(f"{name}: unknown keys {sorted(unknown)}")
            if "values" not in raw:
                raise ConfigurationError(f"{name}: 'values' is required")
            resolution = raw.get("resolution")
            return cls(
                _as_floats(raw["values"], name),
                None if resolution is None else parse_duration(resolution, name=name),
                bool(raw.get("repeat", False)),
            )
        raise ConfigurationError(f"{name}: expected number, list or mapping, got {raw!r}")

    def to_config(self) -> Any:
        """Inverse of :meth:`from_config`."""
        if len(self.values) == 1 and self.repeat:
            return self.values[0]
        config: dict[str, Any] = {"values": list(self.values)}
        if self.resolution is not None:
            config["resolution"] = format_duration(self.resolution)
        if self.repeat:
            config["repeat"] = True
        return config

    def align(self, grid: TimeGrid) -> np.ndarray:
        """Return one value per time step of ``grid``, interpolating linearly if needed."""
        values = np.asarray(self.values, dtype=float)
        data_step_hours = (
            grid.step_hours if self.resolution is None else self.resolution.total_seconds() / 3600
        )
        positions = np.arange(grid.num_steps) * (grid.step_hours / data_step_hours)
        if self.repeat:
            period = len(values)
            wrapped = np.append(values, values[0])
            return np.interp(positions % period, np.arange(period + 1), wrapped)
        if positions[-1] > len(values) - 1:
            raise ConfigurationError(
                f"time series with {len(values)} values does not cover the simulation period; "
                "provide more values or set repeat: true"
            )
        return np.interp(positions, np.arange(len(values)), values)


def align_pandas_series(series: Any, grid: TimeGrid) -> np.ndarray:
    """Linearly interpolate a datetime-indexed ``pandas.Series`` at the steps of ``grid``.

    pandas is optional; this is the only function in ``core.time`` that uses it.
    """
    index = getattr(series, "index", None)
    if index is None or str(getattr(index, "dtype", "")).split("[")[0] != "datetime64":
        raise ConfigurationError("pandas series must have a DatetimeIndex")
    source = index.to_numpy(dtype="datetime64[us]")
    target = grid.datetime64
    if source[0] > target[0] or source[-1] < target[-1]:
        raise ConfigurationError("pandas series does not cover the simulation period")
    return np.interp(
        target.astype(np.int64).astype(float),
        source.astype(np.int64).astype(float),
        series.to_numpy(dtype=float),
    )


def _as_floats(raw: Sequence[Any], name: str) -> tuple[float, ...]:
    try:
        return tuple(float(value) for value in raw)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{name}: all values must be numbers") from error
