"""Charging demand in EV vocabulary, mapped onto :class:`core.Demand`.

Scenario files describe demand with EV terms (``events_per_week``, ``parking_hours``,
``soc_arrival``). This module translates them to the generic arrival processes of
:mod:`core.arrivals` and back, so files stay readable for EV users while the sampling code is
shared with every other application of ``core``.

``model: empirical`` instead resamples measured sessions (:class:`EmpiricalChargingDemand`).
"""

from __future__ import annotations

import csv
import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.arrivals import ArrivalProcess, arrival_process_from_config
from core.config.validation import ProblemCollector
from core.demand import Demand, EmpiricalDemand, place_weekly, reaches_target
from core.errors import ConfigurationError
from core.random import TruncatedNormal
from core.random.samplers import Rng
from core.requests import ServiceProfile
from core.time.grid import TimeGrid
from elvis.infrastructure import ChargingPointSpec
from elvis.vehicle import VehicleType

#: EV model name -> core arrival process name
MODELS = {"weekly": "weekly_profile", "gmm": "gaussian_mixture"}

#: Demand model that resamples measured sessions instead of using an arrival process
EMPIRICAL = "empirical"

#: Where the energy of measured sessions was metered
ENERGY_SIDES = ("grid", "battery")

#: Where session energy comes from: the measured sessions, or vehicle and arrival SOC
ENERGY_FROM = ("sessions", "soc")

#: How sessions without a vehicle get one: by fleet share alone, or among feasible vehicles
VEHICLE_DRAWS = ("independent", "conditional")

#: EV key -> core key, applied to the demand block of a scenario file
KEYS = {
    "model": "process",
    "events_per_week": "arrivals_per_week",
    "arrival_profile": "weekly_profile",
    "parking_hours": "dwell_hours",
    "min_parking_hours": "min_dwell_hours",
    "max_parking_hours": "max_dwell_hours",
}


@dataclass(frozen=True)
class ChargingDemand:
    """When vehicles arrive, how long they park, and how charged they are on arrival.

    Attributes:
        arrivals: Arrival times and parking durations (a core arrival process).
        soc_arrival: State of charge on arrival, clipped to [0, 1].
        soc_target: State of charge at which charging stops; ``None`` means 100 %.
    """

    arrivals: ArrivalProcess
    soc_arrival: TruncatedNormal
    soc_target: TruncatedNormal | None = None

    def for_vehicles(self, vehicles: Sequence[VehicleType]) -> Demand:
        """The core demand model, drawing vehicle types by their probabilities."""
        return Demand(
            self.arrivals,
            self.soc_arrival,
            tuple(vehicles),
            tuple(vehicle.probability for vehicle in vehicles),
            self.soc_target,
        )

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> ChargingDemand:
        if not isinstance(raw, dict) or raw.get("model") not in MODELS:
            model = raw.get("model") if isinstance(raw, dict) else None
            raise ConfigurationError(
                f"demand: model must be one of {sorted([*MODELS, EMPIRICAL])}, got {model!r}"
            )
        if "soc_arrival" not in raw:
            raise ConfigurationError("demand: missing key 'soc_arrival'")
        own = ("soc_arrival", "soc_target")
        arrivals = {KEYS.get(key, key): value for key, value in raw.items() if key not in own}
        arrivals["process"] = MODELS[raw["model"]]
        unknown = [key for key in raw if key not in KEYS and key not in own]
        try:
            process = arrival_process_from_config(arrivals)
        except ConfigurationError as error:
            if unknown:  # report the user's spelling, not the translated one
                raise ConfigurationError(
                    [f"demand: unknown key {key!r}" for key in unknown]
                ) from None
            raise ConfigurationError(_to_ev_terms(error.problems)) from None
        target = raw.get("soc_target")
        return cls(
            process,
            TruncatedNormal.from_config(raw["soc_arrival"], "soc_arrival", low=0, high=1),
            None
            if target is None
            else TruncatedNormal.from_config(target, "soc_target", low=0, high=1),
        )

    def to_config(self) -> dict[str, Any]:
        reverse_keys = {core: ev for ev, core in KEYS.items()}
        reverse_models = {core: ev for ev, core in MODELS.items()}
        config = {
            reverse_keys.get(key, key): value for key, value in self.arrivals.to_config().items()
        }
        config["model"] = reverse_models[config["model"]]
        config = {
            "model": config.pop("model"),
            **config,
            "soc_arrival": self.soc_arrival.to_config(),
        }
        if self.soc_target is not None:
            config["soc_target"] = self.soc_target.to_config()
        return config


@dataclass(frozen=True, slots=True)
class MeasuredSession:
    """One measured charging session.

    Attributes:
        arrival_hour_of_week: Arrival in hours since Monday 00:00, in [0, 168).
        parking_hours: Time the vehicle stayed.
        energy_kwh: Charged energy, grid side (metered) or battery side; see
            :attr:`EmpiricalChargingDemand.energy_side`. Optional (``None``) with
            ``energy_from: soc``.
        vehicle: Vehicle type name (``"brand model"``); ``None`` draws one by probability.
    """

    arrival_hour_of_week: float
    parking_hours: float
    energy_kwh: float | None
    vehicle: str | None = None

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> MeasuredSession:
        ProblemCollector.reject_unknown_keys(
            "session", raw, SESSION_COLUMNS, required=SESSION_COLUMNS[:2]
        )
        vehicle = raw.get("vehicle")
        energy = raw.get("energy_kwh")
        return cls(
            float(raw["arrival_hour_of_week"]),
            float(raw["parking_hours"]),
            None if energy is None or energy == "" else float(energy),
            None if vehicle in (None, "") else str(vehicle),
        )

    def to_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "arrival_hour_of_week": self.arrival_hour_of_week,
            "parking_hours": self.parking_hours,
        }
        if self.energy_kwh is not None:
            config["energy_kwh"] = self.energy_kwh
        if self.vehicle is not None:
            config["vehicle"] = self.vehicle
        return config


#: Keys of an inline session and columns of a session CSV file (``vehicle`` is optional)
SESSION_COLUMNS = ("arrival_hour_of_week", "parking_hours", "energy_kwh", "vehicle")


@dataclass(frozen=True)
class EmpiricalChargingDemand:
    """Resamples measured sessions jointly (bootstrap), as :class:`core.EmpiricalDemand`.

    Arrival hour of week, parking hours, energy and vehicle of a session stay together, so
    all correlations in the data are kept. ``energy_kwh`` is by default the metered grid
    energy (``energy_side: grid``); it is converted to battery energy with the efficiency of
    the session's vehicle (for sessions without a vehicle: of the drawn vehicle), so a fully
    served session draws exactly the measured energy from the grid. With
    ``energy_side: battery`` the energy is used as it is. The SOC on arrival is
    ``1 - battery energy / capacity`` (clipped at 0) and the target SOC is 1. Sessions
    without a vehicle draw one by the vehicle probabilities. The number of sessions is
    Poisson distributed with mean ``events_per_week`` per week.

    Scenario file::

        demand:
          model: empirical
          events_per_week: 120
          sessions:
            - {arrival_hour_of_week: 8.25, parking_hours: 9.0, energy_kwh: 14.2, vehicle: VW ID.3}

    A CSV file has a header with the columns ``arrival_hour_of_week``, ``parking_hours``,
    ``energy_kwh`` and optionally ``vehicle``; a relative path is resolved against the
    scenario file's directory; ``sessions_csv`` can replace ``sessions``.

    Attributes:
        events_per_week: Expected number of sessions per week.
        sessions: The measured sessions.
        sessions_csv: File the sessions were read from, if any; written back by
            :meth:`to_config` instead of the inline sessions.
        energy_side: ``"grid"`` (default, metered energy) or ``"battery"``.
        vehicle_draw: For sessions without a vehicle. ``"independent"`` (default): by the
            vehicle probabilities. ``"conditional"``: by the probabilities restricted to the
            vehicles that can complete the session, i.e. its battery-side energy fits the
            capacity and an uncongested reference point charges it fully within the parking
            time under the engine's physics (:func:`core.demand.reaches_target`: AC limit,
            SOC taper, efficiency). If no vehicle is feasible the session keeps a draw from
            all vehicles and is counted as infeasible (see ``Realisation.draw_stats``).
        energy_from: ``"sessions"`` (default) or ``"soc"``.
        soc_arrival: Arrival SOC distribution; required with ``energy_from: soc``.
        soc_target: Target SOC distribution; ``None`` means 100 %.

    The point a session will use is only known during the simulation, so the reference
    point is the site's charging point if all are equal, otherwise the one with the highest
    rating (its type decides AC or DC). This is an optimistic test on mixed sites.

    With ``energy_from: soc`` the sessions only supply arrival hour and parking hours
    (resampled jointly); their energy and vehicle columns are ignored. The vehicle is drawn
    by probability and the arrival (and target) SOC from ``soc_arrival`` (``soc_target``),
    exactly as in the ``weekly`` model. ``vehicle_draw`` must then be ``independent``.
    """

    events_per_week: float
    sessions: tuple[MeasuredSession, ...]
    sessions_csv: str | None = field(default=None, compare=False)
    energy_side: str = "grid"
    vehicle_draw: str = "independent"
    energy_from: str = "sessions"
    soc_arrival: TruncatedNormal | None = None
    soc_target: TruncatedNormal | None = None

    def __post_init__(self) -> None:
        problems = ProblemCollector("empirical demand")
        problems.non_negative("events_per_week", self.events_per_week)
        if not self.sessions:
            problems.add("needs at least one session")
        if self.energy_side not in ENERGY_SIDES:
            problems.add(f"energy_side must be one of {list(ENERGY_SIDES)}")
        if self.vehicle_draw not in VEHICLE_DRAWS:
            problems.add(f"vehicle_draw must be one of {list(VEHICLE_DRAWS)}")
        if self.energy_from == "soc":
            if self.soc_arrival is None:
                problems.add("energy_from soc needs soc_arrival")
            if self.vehicle_draw != "independent":
                problems.add("energy_from soc needs vehicle_draw independent")
        elif self.energy_from == "sessions":
            if self.soc_arrival is not None or self.soc_target is not None:
                problems.add("soc_arrival and soc_target need energy_from soc")
            if any(s.energy_kwh is None for s in self.sessions):
                problems.add("every session needs energy_kwh (or use energy_from soc)")
        else:
            problems.add(f"energy_from must be one of {list(ENERGY_FROM)}")
        problems.raise_if_any()

    def for_vehicles(
        self, vehicles: Sequence[VehicleType], points: Sequence[ChargingPointSpec] = ()
    ) -> EmpiricalDemand | Demand:
        """The core demand model; vehicle names index into ``vehicles``.

        ``points`` (the site's charging points) are needed for ``vehicle_draw: conditional``.
        """
        if self.energy_from == "soc":
            assert self.soc_arrival is not None
            arrivals = SessionArrivals(
                self.events_per_week,
                np.array([(s.arrival_hour_of_week, s.parking_hours) for s in self.sessions]),
            )
            return Demand(
                arrivals,
                self.soc_arrival,
                tuple(vehicles),
                tuple(vehicle.probability for vehicle in vehicles),
                self.soc_target,
            )
        index = {vehicle.name: i for i, vehicle in enumerate(vehicles)}
        unknown = sorted(
            {s.vehicle for s in self.sessions if s.vehicle is not None and s.vehicle not in index}
        )
        if unknown:
            raise ConfigurationError(
                [f"empirical demand: unknown vehicle {name!r}" for name in unknown]
            )
        rows = [
            (
                s.arrival_hour_of_week,
                s.parking_hours,
                s.energy_kwh,
                np.nan if s.vehicle is None else index[s.vehicle],
            )
            for s in self.sessions
        ]
        return EmpiricalDemand(
            self.events_per_week,
            np.array(rows, dtype=float),
            tuple(vehicles),
            tuple(vehicle.probability for vehicle in vehicles),
            supply_side=self.energy_side == "grid",
            feasible=None
            if self.vehicle_draw == "independent"
            else _feasibility_test(reference_point(points)),
        )

    @classmethod
    def from_config(
        cls, raw: dict[str, Any], base_dir: str | Path | None = None
    ) -> EmpiricalChargingDemand:
        ProblemCollector.reject_unknown_keys(
            "empirical demand",
            raw,
            (
                "model",
                "events_per_week",
                "sessions",
                "sessions_csv",
                "energy_side",
                "vehicle_draw",
                "energy_from",
                "soc_arrival",
                "soc_target",
            ),
            required=("model", "events_per_week"),
        )
        if ("sessions" in raw) == ("sessions_csv" in raw):
            raise ConfigurationError("empirical demand: give either sessions or sessions_csv")
        path = None
        if "sessions_csv" in raw:
            path = Path(raw["sessions_csv"])
            if not path.is_absolute() and base_dir is not None:
                path = Path(base_dir) / path
            sessions = read_sessions_csv(path)
        else:
            sessions = tuple(MeasuredSession.from_config(entry) for entry in raw["sessions"])
        return cls(
            float(raw["events_per_week"]),
            sessions,
            None if path is None else str(path),
            raw.get("energy_side", "grid"),
            raw.get("vehicle_draw", "independent"),
            raw.get("energy_from", "sessions"),
            *(
                None
                if raw.get(key) is None
                else TruncatedNormal.from_config(raw[key], key, low=0, high=1)
                for key in ("soc_arrival", "soc_target")
            ),
        )

    def to_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "model": EMPIRICAL,
            "events_per_week": self.events_per_week,
            "energy_side": self.energy_side,
        }
        if self.vehicle_draw != "independent":
            config["vehicle_draw"] = self.vehicle_draw
        if self.energy_from != "sessions":
            config["energy_from"] = self.energy_from
        for key in ("soc_arrival", "soc_target"):
            value = getattr(self, key)
            if value is not None:
                config[key] = value.to_config()
        if self.sessions_csv is not None:
            config["sessions_csv"] = self.sessions_csv
        else:
            config["sessions"] = [session.to_config() for session in self.sessions]
        return config


@dataclass(frozen=True)
class SessionArrivals:
    """Arrival process that resamples measured ``(hour of week, parking hours)`` pairs.

    Uses the placement of :class:`core.EmpiricalDemand`: a Poisson number of sessions with
    mean ``events_per_week`` per week, each in a random week at its hour of week, rounded up
    to the next time step.
    """

    process = "empirical_sessions"

    events_per_week: float
    pairs: np.ndarray

    def sample(self, grid: TimeGrid, rng: Rng) -> tuple[list[dt.datetime], np.ndarray]:
        week_start, offsets, picked = place_weekly(self.pairs, self.events_per_week, grid, rng)
        arrivals = [week_start + dt.timedelta(hours=float(offset)) for offset in offsets]
        return arrivals, picked[:, 1].astype(float)

    def to_config(self) -> dict[str, Any]:
        return {"process": self.process, "events_per_week": self.events_per_week}


def reference_point(points: Sequence[ChargingPointSpec]) -> ChargingPointSpec:
    """The point used for feasibility: the only kind of point, else the highest rated."""
    if not points:
        raise ConfigurationError("empirical demand: conditional vehicle draw needs points")
    return max(points, key=lambda point: point.max_power_kw)


def _feasibility_test(
    point: ChargingPointSpec,
) -> Callable[[ServiceProfile, float, float, float], bool]:
    def feasible(profile: ServiceProfile, quantity: float, dwell: float, step_hours: float) -> bool:
        return reaches_target(profile, quantity, dwell, step_hours, point.max_power_kw, point.type)

    return feasible


def read_sessions_csv(path: str | Path) -> tuple[MeasuredSession, ...]:
    """Read measured sessions from a CSV file with a header (see :data:`SESSION_COLUMNS`)."""
    try:
        with open(path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    except OSError as error:
        raise ConfigurationError(f"sessions_csv: cannot read {path}: {error}") from None
    try:
        return tuple(MeasuredSession.from_config(row) for row in rows)
    except (ValueError, TypeError) as error:
        raise ConfigurationError(f"sessions_csv {path}: {error}") from None


def demand_from_config(
    raw: dict[str, Any], base_dir: str | Path | None = None
) -> ChargingDemand | EmpiricalChargingDemand:
    """Parse a demand block; ``base_dir`` resolves a relative ``sessions_csv``."""
    if isinstance(raw, dict) and raw.get("model") == EMPIRICAL:
        return EmpiricalChargingDemand.from_config(raw, base_dir)
    return ChargingDemand.from_config(raw)


def _to_ev_terms(problems: list[str]) -> list[str]:
    """Rewrite core terms in error messages into the keys the user actually wrote."""
    replacements = [
        ("weekly_profile arrivals", "weekly demand"),
        ("gaussian_mixture arrivals", "gmm demand"),
        *((core, ev) for ev, core in KEYS.items() if ev != "model"),
    ]
    translated = []
    for problem in problems:
        text = problem
        for core_term, ev_term in replacements:
            text = text.replace(core_term, ev_term)
        translated.append(text)
    return translated
