"""Vehicle types and their traction batteries; a vehicle type is a `core.ServiceProfile`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.config.validation import ProblemCollector


@dataclass(frozen=True, slots=True)
class EVBattery:
    """Charging characteristics of an EV battery.

    Above ``start_power_degradation`` the maximum charging power falls linearly with the state
    of charge (SOC) until it reaches ``max_degradation_level * max_charge_power_kw`` at full
    charge, approximating the constant-voltage phase of lithium-ion charging.

    At an AC charging point the on-board charger limits the power drawn from the grid to
    ``max_ac_power_kw`` (its AC input rating, if given); the battery then receives
    ``max_ac_power_kw x efficiency``. The battery's SOC curve still applies when it binds:
    the grid-side limit is ``min(max_ac_power_kw, max_power_at(soc) / efficiency)``. At a DC
    point (or one of unspecified type) only the SOC curve applies.

    All other power limits refer to the power entering the battery. With ``efficiency`` < 1
    the charging point draws ``power / efficiency`` from the grid.

    Attributes:
        capacity_kwh: Usable energy content.
        max_charge_power_kw: Highest power the battery accepts (the DC limit).
        min_charge_power_kw: Lowest non-zero power the battery accepts.
        efficiency: One-way charging efficiency in (0, 1]: energy stored per energy drawn
            from the grid.
        start_power_degradation: SOC in [0, 1] at which the power starts to fall.
        max_degradation_level: Fraction of the maximum power left at SOC = 1.
        max_ac_power_kw: AC input rating of the on-board charger (grid side); ``None`` means
            no separate AC limit.
        dc_capable: False for vehicles without a DC inlet: they cannot charge at a DC point
            (power 0 there; counted as unservable). Untyped points are unaffected.
    """

    capacity_kwh: float
    max_charge_power_kw: float
    min_charge_power_kw: float = 0.0
    efficiency: float = 1.0
    start_power_degradation: float = 1.0
    max_degradation_level: float = 0.0
    max_ac_power_kw: float | None = None
    dc_capable: bool = True

    def __post_init__(self) -> None:
        problems = ProblemCollector("battery")
        problems.positive("capacity_kwh", self.capacity_kwh)
        problems.positive("max_charge_power_kw", self.max_charge_power_kw)
        problems.non_negative("min_charge_power_kw", self.min_charge_power_kw)
        problems.fraction("efficiency", self.efficiency)
        if self.efficiency == 0:
            problems.add("efficiency must be > 0")
        if self.max_ac_power_kw is not None:
            problems.positive("max_ac_power_kw", self.max_ac_power_kw)
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

    def max_ac_power_at(self, soc: float) -> float:
        """Highest power into the battery at an AC point: on-board charger and SOC curve.

        The charger's grid-side rating is converted to the battery side with ``efficiency``.
        """
        if self.max_ac_power_kw is None:
            return self.max_power_at(soc)
        return min(self.max_ac_power_kw * self.efficiency, self.max_power_at(soc))

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
        """Inverse of :meth:`from_config`; ``max_ac_power_kw`` only if set."""
        config = {name: getattr(self, name) for name in self.__dataclass_fields__}
        if self.max_ac_power_kw is None:
            del config["max_ac_power_kw"]
        if self.dc_capable:
            del config["dc_capable"]
        return config


def _no_power(soc: float) -> float:
    """Rate curve of a vehicle at a point it cannot use."""
    return 0.0


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

    @property
    def efficiency(self) -> float:
        """One-way charging efficiency of the battery."""
        return self.battery.efficiency

    def rate_curve(self, kind: str) -> Callable[[float], float]:
        """Maximum charging power in kW by SOC at a charging point of type ``kind``.

        ``"ac"`` adds the on-board charger limit; ``"dc"`` is 0 without a DC inlet; any other
        type uses :meth:`max_rate_at`.
        """
        if kind == "ac":
            return self.battery.max_ac_power_at
        if kind == "dc" and not self.battery.dc_capable:
            return _no_power
        return self.battery.max_power_at

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
