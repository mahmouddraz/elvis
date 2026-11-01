"""Scenario: a charging site and its users, independent of the simulated period.

A scenario is immutable; derive variants with :meth:`Scenario.replace`. It is a thin, EV-named
layer over :class:`core.System` (see :meth:`Scenario.to_system`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from core.allocation import POLICIES
from core.config.validation import ProblemCollector
from core.config.yaml_io import dump_yaml, load_yaml
from core.errors import ConfigurationError
from core.system import OpenHours, System
from core.system import Operation as CoreOperation
from core.time.series import TimeSeries
from elvis.demand import ChargingDemand
from elvis.infrastructure import InfrastructureSpec
from elvis.vehicle import VehicleType

if TYPE_CHECKING:
    import numpy as np

    from core.time.grid import TimeGrid
    from elvis.realisation import Realisation

SCHEMA_VERSION = 1

DisconnectRule = Literal["at_departure", "when_full"]

#: EV policy name -> (core policy name, {EV parameter: core parameter})
POLICY_NAMES: dict[str, tuple[str, dict[str, str]]] = {
    "uncontrolled": ("unconstrained", {}),
    "fcfs": ("first_come_first_served", {}),
    "edf": ("earliest_departure_first", {}),
    "discrimination_free": ("fair_share", {"charging_period": "block"}),
}

OpeningHours = OpenHours


@dataclass(frozen=True)
class Operation:
    """How the site is operated.

    Attributes:
        scheduling: Policy name or ``{policy: name, **parameters}``. EV names are
            ``uncontrolled``, ``fcfs`` (arrival order), ``edf`` (earliest departure first) and
            ``discrimination_free``
            (parameter ``charging_period``); any :mod:`core.allocation` policy name also works.
        queue_length: Vehicles that may wait for a free charging point (0: no queue).
        disconnect: ``"at_departure"``: vehicles stay until their parking time is over.
            ``"when_full"``: vehicles leave once they reach their target SOC.
        opening_hours: Outside these hours arrivals are rejected and all vehicles leave.
    """

    scheduling: str | Mapping[str, Any] = "uncontrolled"
    queue_length: int = 0
    disconnect: DisconnectRule = "at_departure"
    opening_hours: OpeningHours | None = None

    def __post_init__(self) -> None:
        if self.disconnect not in ("at_departure", "when_full"):
            raise ConfigurationError("operation: disconnect must be 'at_departure' or 'when_full'")
        self.to_core()  # fail early on an unknown policy or bad parameters

    def to_core(self) -> CoreOperation:
        params = (
            {"policy": self.scheduling}
            if isinstance(self.scheduling, str)
            else dict(self.scheduling)
        )
        name = params.pop("policy", None)
        if name in POLICY_NAMES:
            name, renames = POLICY_NAMES[name]
            params = {renames.get(key, key): value for key, value in params.items()}
        elif name not in POLICIES:
            choices = sorted(POLICY_NAMES) + sorted(POLICIES)
            raise ConfigurationError(f"scheduling: policy must be one of {choices}, got {name!r}")
        return CoreOperation(
            policy={"policy": name, **params},
            queue_capacity=self.queue_length,
            release="at_departure" if self.disconnect == "at_departure" else "at_target",
            open_hours=self.opening_hours,
        )

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> Operation:
        ProblemCollector.reject_unknown_keys(
            "operation", raw, ("scheduling", "queue_length", "disconnect", "opening_hours")
        )
        hours = raw.get("opening_hours")
        if hours is not None and (len(hours) != 2 or not 0 <= hours[0] <= hours[1] <= 24):
            raise ConfigurationError(
                "opening_hours: need [opens, closes] with 0 <= opens <= closes <= 24"
            )
        return cls(
            scheduling=raw.get("scheduling", "uncontrolled"),
            queue_length=int(raw.get("queue_length", 0)),
            disconnect=raw.get("disconnect", "at_departure"),
            opening_hours=None if hours is None else OpeningHours(*hours),
        )

    def to_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "scheduling": self.scheduling
            if isinstance(self.scheduling, str)
            else dict(self.scheduling),
            "queue_length": self.queue_length,
            "disconnect": self.disconnect,
        }
        if self.opening_hours is not None:
            config["opening_hours"] = [self.opening_hours.opens, self.opening_hours.closes]
        return config


@dataclass(frozen=True)
class Scenario:
    """A charging site and its users.

    Attributes:
        name: Short identifier, e.g. ``"office"``.
        infrastructure: Transformers, stations and charging points.
        vehicles: Vehicle types that may arrive.
        demand: Arrival times, parking durations and arrival SOC.
        operation: Scheduling, queueing and opening hours.
        transformer_preload: Other load on the transformer in kW (default: none).
        emission_factors: Grid emission factor in g CO2 per kWh (default: not tracked).
        description: Free text.
    """

    name: str
    infrastructure: InfrastructureSpec
    vehicles: tuple[VehicleType, ...]
    demand: ChargingDemand
    operation: Operation = field(default_factory=Operation)
    transformer_preload: TimeSeries | None = None
    emission_factors: TimeSeries | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.vehicles:
            raise ConfigurationError(f"scenario {self.name!r}: needs at least one vehicle type")
        names = [vehicle.name for vehicle in self.vehicles]
        if len(set(names)) != len(names):
            raise ConfigurationError(f"scenario {self.name!r}: vehicle names must be unique")

    @cached_property
    def system(self) -> System:
        """The equivalent domain-agnostic :class:`core.System`."""
        return System(
            network=self.infrastructure.to_network(),
            demand=self.demand.for_vehicles(self.vehicles),
            operation=self.operation.to_core(),
            background=self.transformer_preload,
        )

    def replace(self, **changes: Any) -> Scenario:
        """A copy with some attributes changed (the original is left untouched)."""
        return dataclasses.replace(self, **changes)

    def realise(self, grid: TimeGrid, seed: int | np.random.Generator | None = None) -> Realisation:
        """Sample one set of charging events for ``grid``; ``seed`` may be a generator."""
        from elvis.realisation import Realisation  # noqa: PLC0415 - realisation imports scenario

        return Realisation(self, self.system.realise(grid, seed))

    # -- serialisation ---------------------------------------------------------------------

    @classmethod
    def from_config(cls, raw: dict[str, Any], *, source: str | None = None) -> Scenario:
        """Build from a parsed scenario file. All problems are reported together."""
        allowed = (
            "schema_version",
            "name",
            "description",
            "infrastructure",
            "vehicles",
            "demand",
            "operation",
            "transformer_preload",
            "emission_factors",
        )
        try:
            ProblemCollector.reject_unknown_keys(
                "scenario", raw, allowed, required=("name", "infrastructure", "vehicles", "demand")
            )
            version = raw.get("schema_version", SCHEMA_VERSION)
            if version != SCHEMA_VERSION:
                raise ConfigurationError(f"unsupported schema_version {version}")
            return cls(
                name=str(raw["name"]),
                description=str(raw.get("description", "")),
                infrastructure=InfrastructureSpec.from_config(raw["infrastructure"]),
                vehicles=tuple(VehicleType.from_config(v) for v in raw["vehicles"]),
                demand=ChargingDemand.from_config(raw["demand"]),
                operation=Operation.from_config(raw.get("operation", {})),
                transformer_preload=TimeSeries.from_config(
                    raw.get("transformer_preload"), name="transformer_preload"
                ),
                emission_factors=TimeSeries.from_config(
                    raw.get("emission_factors"), name="emission_factors"
                ),
            )
        except ConfigurationError as error:
            raise ConfigurationError(error.problems, source=source) from None

    def to_config(self) -> dict[str, Any]:
        """Inverse of :meth:`from_config`."""
        config: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "operation": self.operation.to_config(),
            "infrastructure": self.infrastructure.to_config(),
            "demand": self.demand.to_config(),
            "vehicles": [vehicle.to_config() for vehicle in self.vehicles],
        }
        if self.transformer_preload is not None:
            config["transformer_preload"] = self.transformer_preload.to_config()
        if self.emission_factors is not None:
            config["emission_factors"] = self.emission_factors.to_config()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> Scenario:
        """Load a scenario file."""
        return cls.from_config(load_yaml(path), source=str(path))

    def to_yaml(self, path: str | Path) -> None:
        """Write a scenario file that :meth:`from_yaml` reads back unchanged."""
        dump_yaml(self.to_config(), path)
