"""Vehicle types and their traction batteries; a vehicle type is a `core.ServiceProfile`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.config.validation import ProblemCollector


@dataclass(frozen=True, slots=True)
class EVBattery:
    """Charging characteristics of an EV battery.

    Above ``start_power_degradation`` the maximum charging power falls linearly with the state
    of charge (SOC) until it reaches ``max_degradation_level * max_charge_power_kw`` at full
    charge, approximating the constant-voltage phase of lithium-ion charging.

    Attributes:
        capacity_kwh: Usable energy content.
        max_charge_power_kw: Highest power the battery accepts.
        min_charge_power_kw: Lowest non-zero power the battery accepts.
        efficiency: Charging efficiency in [0, 1]. Stored for reporting; the charging model
            does not apply losses yet.
        start_power_degradation: SOC in [0, 1] at which the power starts to fall.
        max_degradation_level: Fraction of the maximum power left at SOC = 1.
    """

    capacity_kwh: float
    max_charge_power_kw: float
    min_charge_power_kw: float = 0.0
    efficiency: float = 1.0
    start_power_degradation: float = 1.0
    max_degradation_level: float = 0.0

    def __post_init__(self) -> None:
        problems = ProblemCollector("battery")
        problems.positive("capacity_kwh", self.capacity_kwh)
        problems.positive("max_charge_power_kw", self.max_charge_power_kw)
        problems.non_negative("min_charge_power_kw", self.min_charge_power_kw)
        problems.fraction("efficiency", self.efficiency)
        problems.fraction("start_power_degradation", self.start_power_degradation)
        problems.fraction("max_degradation_level", self.max_degradation_level)
        if self.min_charge_power_kw > self.max_charge_power_kw:
            problems.add("min_charge_power_kw exceeds max_charge_power_kw")
        problems.raise_if_any()

    def max_power_at(self, soc: float) -> float:
        """Highest charging power at the given state of charge."""
        if soc <= self.start_power_degradation:
            return self.max_charge_power_kw
        degradation = (
            (soc - self.start_power_degradation)
            / (1 - self.start_power_degradation)
            * self.max_charge_power_kw
            * (1 - self.max_degradation_level)
        )
        return self.max_charge_power_kw - degradation

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> EVBattery:
        """Build from a scenario-file mapping; unknown keys are an error."""
        ProblemCollector.reject_unknown_keys(
            "battery",
            raw,
            cls.__dataclass_fields__,
            required=("capacity_kwh", "max_charge_power_kw"),
        )
        return cls(**raw)

    def to_config(self) -> dict[str, Any]:
        """Inverse of :meth:`from_config`."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class VehicleType:
    """A vehicle model and its share of the arriving fleet.

    Attributes:
        brand: Manufacturer, e.g. ``"VW"``.
        model: Model name, e.g. ``"ID.3"``.
        battery: Battery characteristics.
        probability: Relative frequency among arriving vehicles. Weights are normalised over
            all vehicle types of a scenario, so they need not sum to one.
    """

    brand: str
    model: str
    battery: EVBattery = field(repr=False)
    probability: float

    def __post_init__(self) -> None:
        if self.probability < 0:
            raise ProblemCollector.single(f"vehicle {self.name}: probability must be >= 0")

    @property
    def name(self) -> str:
        """Human-readable ``"brand model"``."""
        return f"{self.brand} {self.model}"

    def __str__(self) -> str:
        return self.name

    # -- core.ServiceProfile: the vehicle's fill level is its state of charge --------------

    @property
    def capacity(self) -> float:
        """Battery capacity in kWh."""
        return self.battery.capacity_kwh

    def max_rate_at(self, level: float) -> float:
        """Maximum charging power in kW at state of charge ``level``."""
        return self.battery.max_power_at(level)

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> VehicleType:
        """Build from a scenario-file mapping; unknown keys are an error."""
        ProblemCollector.reject_unknown_keys(
            "vehicle type",
            raw,
            cls.__dataclass_fields__,
            required=cls.__dataclass_fields__,
        )
        return cls(
            brand=str(raw["brand"]),
            model=str(raw["model"]),
            battery=EVBattery.from_config(raw["battery"]),
            probability=float(raw["probability"]),
        )

    def to_config(self) -> dict[str, Any]:
        """Inverse of :meth:`from_config`."""
        return {
            "brand": self.brand,
            "model": self.model,
            "probability": self.probability,
            "battery": self.battery.to_config(),
        }
