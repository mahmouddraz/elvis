"""Requests: units of demand that arrive, wait for or occupy a server, and leave.

A request carries a *fill level* in [0, 1] that service raises towards a target, at a rate
bounded by its :class:`ServiceProfile`. For an electric vehicle the level is the state of
charge and the profile its battery; for a hydrogen car it would be the tank pressure, for a
compute job its progress.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.errors import ConfigurationError


@runtime_checkable
class ServiceProfile(Protocol):
    """How a kind of request accepts service.

    Attributes:
        name: Unique name of the request kind (used in saved realisations).
        capacity: Quantity that raises the level from 0 to 1 (e.g. kWh).
    """

    @property
    def name(self) -> str: ...

    @property
    def capacity(self) -> float: ...

    def max_rate_at(self, level: float) -> float:
        """Highest service rate (quantity per hour) accepted at ``level``."""
        ...


@dataclass(frozen=True, slots=True)
class Request:
    """One arrival, as sampled from a demand model or taken from measured data.

    Attributes:
        id: Unique, increasing in order of arrival within one realisation.
        arrival: When the request arrives.
        dwell_hours: How long it may stay in the system.
        initial_level: Fill level on arrival, in [0, 1].
        profile: How it accepts service.
        target_level: Service stops at this level.
        departure: ``arrival + dwell_hours``.
    """

    id: int
    arrival: dt.datetime
    dwell_hours: float
    initial_level: float
    profile: ServiceProfile
    target_level: float = 1.0
    departure: dt.datetime = field(init=False, compare=False)

    def __post_init__(self) -> None:
        if self.dwell_hours < 0:
            raise ConfigurationError(f"request {self.id}: dwell_hours must be >= 0")
        if not 0 <= self.initial_level <= 1 or not 0 <= self.target_level <= 1:
            raise ConfigurationError(f"request {self.id}: levels must be in [0, 1]")
        departure = self.arrival + dt.timedelta(hours=self.dwell_hours)
        object.__setattr__(self, "departure", departure)

    @property
    def demand(self) -> float:
        """Quantity needed to reach the target level."""
        return max(self.target_level - self.initial_level, 0) * self.profile.capacity

    def to_record(self) -> dict[str, Any]:
        """Flat, JSON-compatible representation (the profile is referenced by name)."""
        return {
            "id": self.id,
            "arrival": self.arrival.isoformat(),
            "dwell_hours": self.dwell_hours,
            "initial_level": self.initial_level,
            "target_level": self.target_level,
            "profile": self.profile.name,
        }

    @classmethod
    def from_record(
        cls, record: dict[str, Any], profiles_by_name: Mapping[str, ServiceProfile]
    ) -> Request:
        """Inverse of :meth:`to_record`."""
        try:
            profile = profiles_by_name[record["profile"]]
        except KeyError as error:
            raise ConfigurationError(f"request {record.get('id')}: unknown profile") from error
        return cls(
            id=int(record["id"]),
            arrival=dt.datetime.fromisoformat(record["arrival"]),
            dwell_hours=float(record["dwell_hours"]),
            initial_level=float(record["initial_level"]),
            profile=profile,
            target_level=float(record.get("target_level", 1.0)),
        )
