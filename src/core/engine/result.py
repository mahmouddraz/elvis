"""Simulation output and generic indicators.

Rates are constant within a time step; quantities are rates integrated over time
(rate x hours). Units are whatever the application uses (e.g. kW and kWh).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from core.errors import ConfigurationError
from core.time.grid import HOURS_PER_DAY, hour_of_day

if TYPE_CHECKING:
    from core.system import Realisation


@dataclass(slots=True)
class SessionRecord:
    """When a request was served: from its first to its last step with a rate > 0."""

    request_id: int
    server_id: str
    first_served_step: int
    last_served_step: int


class Result:
    """Rate time series and indicators of one simulation run.

    Attributes:
        realisation: The simulated inputs.
        server_ids: Column labels of :attr:`rates`.
        buffer_ids: Column labels of :attr:`buffer_rates`.
        rates: Array ``(steps, servers)`` of service rates.
        buffer_rates: Array ``(steps, buffers)``; positive values fill, negative drain.
        rejections: Arrivals that found neither a free server nor queue space.
        sessions: Service interval of every request that was served.
        delivered: Quantity delivered per request id, for every request that was connected
            to a server (zero if it never received a rate).
    """

    def __init__(
        self,
        realisation: Realisation,
        server_ids: list[str],
        buffer_ids: list[str],
        rates: np.ndarray,
        buffer_rates: np.ndarray,
        rejections: int,
        sessions: list[SessionRecord],
        delivered: dict[int, float] | None = None,
    ) -> None:
        self.realisation = realisation
        self.server_ids = server_ids
        self.buffer_ids = buffer_ids
        self.rates = rates
        self.buffer_rates = buffer_rates
        self.rejections = rejections
        self.sessions = sessions
        self.delivered = delivered or {}

    @property
    def step_hours(self) -> float:
        return self.realisation.grid.step_hours

    # -- time series -----------------------------------------------------------------------

    @cached_property
    def total_rate(self) -> np.ndarray:
        """Sum of all server rates per step."""
        return self.rates.sum(axis=1)

    @cached_property
    def total_buffer_rate(self) -> np.ndarray:
        """Sum of all buffer rates per step (positive: filling)."""
        return self.buffer_rates.sum(axis=1)

    @cached_property
    def root_load(self) -> np.ndarray:
        """Load on the root node(s): servers plus buffers plus background."""
        return self.total_rate + self.total_buffer_rate + self.realisation.background()

    def to_dataframe(self) -> Any:
        """Per-server and aggregate rates indexed by time (requires pandas)."""
        try:
            import pandas as pd  # noqa: PLC0415 - optional dependency
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError(
                "to_dataframe() needs pandas: pip install 'elvislis[pandas]'"
            ) from error
        index = pd.DatetimeIndex(self.realisation.grid.datetime64, name="time")
        frame = pd.DataFrame(self.rates, index=index, columns=self.server_ids)
        for column, buffer_id in enumerate(self.buffer_ids):
            frame[buffer_id] = self.buffer_rates[:, column]
        frame["total"] = self.total_rate
        frame["root_load"] = self.root_load
        return frame

    # -- indicators ------------------------------------------------------------------------

    @property
    def num_requests(self) -> int:
        return len(self.realisation.requests)

    @property
    def rejection_rate(self) -> float:
        """Share of arrivals that could not be served at all."""
        return self.rejections / self.num_requests if self.num_requests else 0.0

    @property
    def total_quantity(self) -> float:
        """Quantity delivered to requests."""
        return float(self.total_rate.sum() * self.step_hours)

    @property
    def peak_rate(self) -> float:
        """Highest total server rate."""
        return float(self.total_rate.max(initial=0.0))

    @property
    def nominal_rate(self) -> float:
        """Sum of the maximum rate of all servers."""
        return self.realisation.system.network.nominal_rate

    def utilisation(self, quantile: float | None = None) -> float:
        """Total rate relative to the nominal rate: the peak, or a quantile over time."""
        ratio = self.total_rate / self.nominal_rate
        return float(ratio.max() if quantile is None else np.quantile(ratio, quantile))

    def utilisation_histogram(
        self, bins: int | Sequence[float] = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(counts, bin_edges)`` of the per-step utilisation."""
        return np.histogram(self.total_rate / self.nominal_rate, bins)

    def request_outcomes(self) -> dict[str, np.ndarray]:
        """Per request, in arrival order: ``demand``, ``delivered`` and ``connected``.

        ``demand`` is the quantity needed to reach the target level; ``connected`` is False
        for requests that were rejected or gave up while waiting.
        """
        requests = self.realisation.requests
        return {
            "demand": np.array([r.demand for r in requests]),
            "delivered": np.array([self.delivered.get(r.id, 0.0) for r in requests]),
            "connected": np.array([r.id in self.delivered for r in requests]),
        }

    def service_hours(self) -> np.ndarray:
        """Hours from the start of the first to the end of the last served step per request."""
        steps = np.array([s.last_served_step - s.first_served_step + 1 for s in self.sessions])
        return steps * self.step_hours

    def mean_service_hours(self) -> float:
        """Average of :meth:`service_hours` (NaN without sessions)."""
        hours = self.service_hours()
        return float(hours.mean()) if len(hours) else float("nan")

    def weighted_quantity(self, weights: float | np.ndarray) -> float:
        """Delivered quantity weighted by e.g. a price or an emission intensity.

        Args:
            weights: A constant, or one value per time step (see :func:`daily_curve`).
        """
        quantity = self.total_rate * self.step_hours
        if np.isscalar(weights):
            return float(quantity.sum() * float(weights))  # type: ignore[arg-type]
        values = np.asarray(weights, dtype=float)
        if values.shape != quantity.shape:
            raise ConfigurationError(f"need one weight per time step ({len(quantity)})")
        return float((quantity * values).sum())

    def daily_curve(self, values: Sequence[float]) -> np.ndarray:
        """Per-step values from a daily curve repeated every day of the simulation.

        ``values`` are equally spaced from midnight and interpolated linearly, wrapping at
        midnight (e.g. 24 hourly electricity prices).
        """
        curve = np.asarray(values, dtype=float)
        if curve.ndim != 1 or len(curve) == 0:
            raise ConfigurationError("daily curve must be a non-empty list")
        hours = np.array([hour_of_day(ts) for ts in self.realisation.grid.timestamps])
        knots = np.arange(len(curve) + 1) * (HOURS_PER_DAY / len(curve))
        return np.interp(hours, knots, np.append(curve, curve[0]))

    def summary(self) -> dict[str, Any]:
        """The headline indicators as a flat dictionary."""
        return {
            "requests": self.num_requests,
            "rejections": self.rejections,
            "rejection_rate": self.rejection_rate,
            "total_quantity": self.total_quantity,
            "peak_rate": self.peak_rate,
            "utilisation": self.utilisation(),
            "mean_service_hours": self.mean_service_hours(),
        }
