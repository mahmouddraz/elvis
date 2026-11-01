"""Charging infrastructure in EV vocabulary, mapped onto a :class:`core.NetworkSpec`.

The specification is an immutable tree::

    InfrastructureSpec
    └── TransformerSpec      -> core root node (grid connection; storage -> buffer)
        └── ChargingStationSpec  -> core inner node (shared power electronics)
            └── ChargingPointSpec    -> core server (one connector, one vehicle at a time)

Rates are powers in kW, buffer quantities energies in kWh.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from core.config.validation import ProblemCollector
from core.errors import ConfigurationError
from core.network.spec import BufferSpec, NetworkSpec, NodeSpec, ServerSpec


@dataclass(frozen=True, slots=True)
class ChargingPointSpec:
    """A single connector.

    Attributes:
        max_power_kw: Highest power the connector delivers.
        min_power_kw: Lowest non-zero power the connector delivers.
    """

    max_power_kw: float
    min_power_kw: float = 0.0

    def __post_init__(self) -> None:
        _check_power_limits("charging point", self.min_power_kw, self.max_power_kw)

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> ChargingPointSpec:
        _reject_unknown("charging point", raw, ("max_power_kw", "min_power_kw"))
        return cls(**raw)

    def to_config(self) -> dict[str, Any]:
        return _limits_config(self)


@dataclass(frozen=True, slots=True)
class ChargingStationSpec:
    """Power electronics shared by one or more charging points.

    Attributes:
        max_power_kw: Highest total power of all its charging points.
        charging_points: The connectors of this station.
        min_power_kw: Lowest non-zero power.
    """

    max_power_kw: float
    charging_points: tuple[ChargingPointSpec, ...]
    min_power_kw: float = 0.0

    def __post_init__(self) -> None:
        _check_power_limits("charging station", self.min_power_kw, self.max_power_kw)
        if not self.charging_points:
            raise ConfigurationError("charging station: needs at least one charging point")

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> list[ChargingStationSpec]:
        """Parse one entry; ``count: n`` expands to ``n`` identical stations."""
        _reject_unknown(
            "charging station",
            raw,
            ("max_power_kw", "min_power_kw", "charging_points", "count"),
            required=("max_power_kw", "charging_points"),
        )
        count = int(raw.get("count", 1))
        if count < 1:
            raise ConfigurationError("charging station: count must be >= 1")
        points = _expand_points(raw["charging_points"])
        station = cls(raw["max_power_kw"], points, raw.get("min_power_kw", 0.0))
        return [station] * count

    def to_config(self, count: int = 1) -> dict[str, Any]:
        config = _limits_config(self)
        if count > 1:
            config["count"] = count
        config["charging_points"] = _compact_points(self.charging_points)
        return config


@dataclass(frozen=True, slots=True)
class StorageSpec:
    """A stationary battery behind the transformer that can buffer peaks.

    Attributes:
        capacity_kwh: Usable energy content.
        max_power_kw: Highest (dis)charging power.
        min_power_kw: Lowest non-zero (dis)charging power.
        efficiency: Round-trip efficiency in (0, 1], applied half on charging, half on discharging.
        initial_soc: State of charge at the start of the simulation.
        min_soc: The storage is never discharged below this state of charge.
    """

    capacity_kwh: float
    max_power_kw: float
    min_power_kw: float = 0.0
    efficiency: float = 1.0
    initial_soc: float = 0.0
    min_soc: float = 0.0

    def __post_init__(self) -> None:
        problems = ProblemCollector("storage")
        problems.positive("capacity_kwh", self.capacity_kwh)
        problems.fraction("efficiency", self.efficiency)
        problems.fraction("initial_soc", self.initial_soc)
        problems.fraction("min_soc", self.min_soc)
        if self.min_soc > self.initial_soc:
            problems.add("min_soc exceeds initial_soc")
        problems.raise_if_any()
        _check_power_limits("storage", self.min_power_kw, self.max_power_kw)

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> StorageSpec:
        _reject_unknown(
            "storage", raw, cls.__dataclass_fields__, required=("capacity_kwh", "max_power_kw")
        )
        return cls(**raw)

    def to_config(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class TransformerSpec:
    """The grid connection point.

    Attributes:
        max_power_kw: Highest power drawn from the grid, including the preload.
        charging_stations: Stations fed by this transformer.
        min_power_kw: Lowest non-zero power.
        storage: Optional stationary battery.
    """

    max_power_kw: float
    charging_stations: tuple[ChargingStationSpec, ...]
    min_power_kw: float = 0.0
    storage: StorageSpec | None = None

    def __post_init__(self) -> None:
        _check_power_limits("transformer", self.min_power_kw, self.max_power_kw)
        if not self.charging_stations:
            raise ConfigurationError("transformer: needs at least one charging station")

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> TransformerSpec:
        _reject_unknown(
            "transformer",
            raw,
            ("max_power_kw", "min_power_kw", "charging_stations", "storage"),
            required=("max_power_kw", "charging_stations"),
        )
        stations = tuple(
            station
            for entry in raw["charging_stations"]
            for station in ChargingStationSpec.from_config(entry)
        )
        storage = raw.get("storage")
        return cls(
            raw["max_power_kw"],
            stations,
            raw.get("min_power_kw", 0.0),
            None if storage is None else StorageSpec.from_config(storage),
        )

    def to_config(self) -> dict[str, Any]:
        config = _limits_config(self)
        config["charging_stations"] = [
            station.to_config(count) for station, count in _run_lengths(self.charging_stations)
        ]
        if self.storage is not None:
            config["storage"] = self.storage.to_config()
        return config


@dataclass(frozen=True, slots=True)
class InfrastructureSpec:
    """All transformers of a site."""

    transformers: tuple[TransformerSpec, ...]

    def __post_init__(self) -> None:
        if not self.transformers:
            raise ConfigurationError("infrastructure: needs at least one transformer")

    @property
    def charging_points(self) -> list[ChargingPointSpec]:
        """All charging points, in network order."""
        return [
            point
            for transformer in self.transformers
            for station in transformer.charging_stations
            for point in station.charging_points
        ]

    @property
    def nominal_power_kw(self) -> float:
        """Sum of the maximum power of all charging points."""
        return sum(point.max_power_kw for point in self.charging_points)

    def to_network(self) -> NetworkSpec:
        """The equivalent core network, with ids ``transformer-1``, ``station-1``, ``cp-1``..."""
        roots = []
        point_number = 0
        for t, transformer in enumerate(self.transformers, start=1):
            prefix = "" if len(self.transformers) == 1 else f"t{t}-"
            stations = []
            for s, station in enumerate(transformer.charging_stations, start=1):
                servers = []
                for point in station.charging_points:
                    point_number += 1
                    servers.append(
                        ServerSpec(point.max_power_kw, point.min_power_kw, f"cp-{point_number}")
                    )
                stations.append(
                    NodeSpec(
                        station.max_power_kw,
                        tuple(servers),
                        station.min_power_kw,
                        id=f"{prefix}station-{s}",
                    )
                )
            storage = transformer.storage
            buffer = None
            if storage is not None:
                buffer = BufferSpec(
                    storage.capacity_kwh,
                    storage.max_power_kw,
                    storage.min_power_kw,
                    storage.efficiency,
                    storage.initial_soc,
                    storage.min_soc,
                    id=f"storage-{t}",
                )
            roots.append(
                NodeSpec(
                    transformer.max_power_kw,
                    tuple(stations),
                    transformer.min_power_kw,
                    buffer,
                    id=f"transformer-{t}",
                )
            )
        return NetworkSpec(tuple(roots))

    @classmethod
    def wallboxes(
        cls,
        num_points: int,
        point_power_kw: float,
        *,
        points_per_station: int = 1,
        station_power_kw: float | None = None,
        transformer_power_kw: float | None = None,
        storage: StorageSpec | None = None,
    ) -> InfrastructureSpec:
        """A homogeneous site; unspecified limits default to the sum of the points below."""
        if num_points < 1 or points_per_station < 1 or num_points % points_per_station:
            raise ConfigurationError("num_points must be a positive multiple of points_per_station")
        point = ChargingPointSpec(point_power_kw)
        station = ChargingStationSpec(
            station_power_kw or point_power_kw * points_per_station,
            (point,) * points_per_station,
        )
        transformer = TransformerSpec(
            transformer_power_kw or point_power_kw * num_points,
            (station,) * (num_points // points_per_station),
            storage=storage,
        )
        return cls((transformer,))

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> InfrastructureSpec:
        _reject_unknown("infrastructure", raw, ("transformers",), required=("transformers",))
        return cls(tuple(TransformerSpec.from_config(entry) for entry in raw["transformers"]))

    def to_config(self) -> dict[str, Any]:
        return {"transformers": [transformer.to_config() for transformer in self.transformers]}


def _check_power_limits(context: str, min_power_kw: float, max_power_kw: float) -> None:
    problems = ProblemCollector(context)
    problems.positive("max_power_kw", max_power_kw)
    problems.non_negative("min_power_kw", min_power_kw)
    if not problems.problems and min_power_kw > max_power_kw:
        problems.add("min_power_kw exceeds max_power_kw")
    problems.raise_if_any()


def _reject_unknown(
    context: str, raw: Any, allowed: Collection[str], *, required: Collection[str] = ()
) -> None:
    ProblemCollector.reject_unknown_keys(
        context, raw, allowed, required=required or ("max_power_kw",)
    )


def _limits_config(spec: Any) -> dict[str, Any]:
    config: dict[str, Any] = {"max_power_kw": spec.max_power_kw}
    if spec.min_power_kw:
        config["min_power_kw"] = spec.min_power_kw
    return config


def _expand_points(raw_points: list[dict[str, Any]]) -> tuple[ChargingPointSpec, ...]:
    points: list[ChargingPointSpec] = []
    for raw in raw_points:
        fields = {key: value for key, value in raw.items() if key != "count"}
        points += [ChargingPointSpec.from_config(fields)] * int(raw.get("count", 1))
    return tuple(points)


def _compact_points(points: Sequence[ChargingPointSpec]) -> list[dict[str, Any]]:
    compacted = []
    for point, count in _run_lengths(points):
        config = point.to_config()
        if count > 1:
            config["count"] = count
        compacted.append(config)
    return compacted


def _run_lengths(items: Sequence[Any]) -> list[tuple[Any, int]]:
    runs: list[tuple[Any, int]] = []
    for item in items:
        if runs and runs[-1][0] == item:
            runs[-1] = (item, runs[-1][1] + 1)
        else:
            runs.append((item, 1))
    return runs
