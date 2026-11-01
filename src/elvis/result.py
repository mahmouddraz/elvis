"""Charging-site results and KPIs, in kW and kWh, on top of :class:`core.Result`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from core.engine.result import Result, SessionRecord
from core.errors import ConfigurationError
from elvis.realisation import Realisation


class ChargingResult:
    """Power time series and KPIs of one charging simulation.

    Powers are in kW and constant within a time step; energies in kWh. The generic result is
    available as :attr:`core` for anything not wrapped here.
    """

    def __init__(self, realisation: Realisation, core: Result) -> None:
        self.realisation = realisation
        self.core = core

    def __repr__(self) -> str:
        return (
            f"ChargingResult({self.realisation.scenario.name!r}, "
            f"{self.realisation.grid.num_steps} steps, peak {self.peak_load_kw:.1f} kW)"
        )

    # -- time series -----------------------------------------------------------------------

    @property
    def point_ids(self) -> list[str]:
        return self.core.server_ids

    @property
    def storage_ids(self) -> list[str]:
        return self.core.buffer_ids

    @property
    def point_power_kw(self) -> np.ndarray:
        """Array ``(steps, charging points)`` of charging power."""
        return self.core.rates

    @property
    def storage_power_kw(self) -> np.ndarray:
        """Array ``(steps, storage systems)``; positive values charge the storage."""
        return self.core.buffer_rates

    @property
    def load_profile_kw(self) -> np.ndarray:
        """Total charging power of all charging points per step."""
        return self.core.total_rate

    @property
    def storage_profile_kw(self) -> np.ndarray:
        """Total storage power per step (positive: charging)."""
        return self.core.total_buffer_rate

    @property
    def grid_profile_kw(self) -> np.ndarray:
        """Power drawn from the grid: charging plus storage plus preload."""
        return self.core.root_load

    def to_dataframe(self) -> Any:
        """Per-point and aggregate power indexed by time (requires pandas)."""
        return self.core.to_dataframe().rename(
            columns={"total": "charging_total", "root_load": "grid_total"}
        )

    # -- KPIs ------------------------------------------------------------------------------

    @property
    def sessions(self) -> list[SessionRecord]:
        return self.core.sessions

    @property
    def num_events(self) -> int:
        return self.core.num_requests

    @property
    def rejections(self) -> int:
        return self.core.rejections

    @property
    def rejection_rate(self) -> float:
        """Share of arriving vehicles that could not be served at all."""
        return self.core.rejection_rate

    @property
    def total_energy_kwh(self) -> float:
        """Energy delivered to vehicles."""
        return self.core.total_quantity

    @property
    def peak_load_kw(self) -> float:
        """Highest total charging power."""
        return self.core.peak_rate

    @property
    def nominal_power_kw(self) -> float:
        """Sum of the rated power of all charging points."""
        return self.core.nominal_rate

    def simultaneity_factor(self, quantile: float | None = None) -> float:
        """Charging power relative to the nominal power: the peak, or a quantile over time."""
        return self.core.utilisation(quantile)

    def simultaneity_histogram(
        self, bins: int | Sequence[float] = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(counts, bin_edges)`` of the per-step simultaneity factor."""
        return self.core.utilisation_histogram(bins)

    def charging_hours(self) -> np.ndarray:
        """Hours from the start of the first to the end of the last charging step per session."""
        return self.core.service_hours()

    def mean_charging_hours(self) -> float:
        """Average of :meth:`charging_hours` (NaN without sessions)."""
        return self.core.mean_service_hours()

    def energy_cost(self, price_per_kwh: float | Sequence[float]) -> float:
        """Cost of the charging energy.

        Args:
            price_per_kwh: A constant price, or a daily price curve of equally spaced values
                from midnight (e.g. 24 hourly prices), interpolated linearly.
        """
        if np.isscalar(price_per_kwh):
            return self.core.weighted_quantity(float(price_per_kwh))  # type: ignore[arg-type]
        return self.core.weighted_quantity(self.core.daily_curve(price_per_kwh))  # type: ignore[arg-type]

    def emissions_kg(self) -> float:
        """CO2 emissions of the charging energy from the scenario's emission factors."""
        factors = self.realisation.emission_factors()
        if factors is None:
            raise ConfigurationError("the scenario defines no emission_factors")
        return self.core.weighted_quantity(factors) / 1000

    def summary(self) -> dict[str, Any]:
        """The headline KPIs as a flat dictionary."""
        return {
            "scenario": self.realisation.scenario.name,
            "events": self.num_events,
            "rejections": self.rejections,
            "rejection_rate": self.rejection_rate,
            "energy_kwh": self.total_energy_kwh,
            "peak_load_kw": self.peak_load_kw,
            "simultaneity_factor": self.simultaneity_factor(),
            "mean_charging_hours": self.mean_charging_hours(),
        }
