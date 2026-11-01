"""Charging demand in EV vocabulary, mapped onto :class:`core.Demand`.

Scenario files describe demand with EV terms (``events_per_week``, ``parking_hours``,
``soc_arrival``). This module translates them to the generic arrival processes of
:mod:`core.arrivals` and back, so files stay readable for EV users while the sampling code is
shared with every other application of ``core``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from core.arrivals import ArrivalProcess, arrival_process_from_config
from core.demand import Demand
from core.errors import ConfigurationError
from core.random import TruncatedNormal
from elvis.vehicle import VehicleType

#: EV model name -> core arrival process name
MODELS = {"weekly": "weekly_profile", "gmm": "gaussian_mixture"}

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
                f"demand: model must be one of {sorted(MODELS)}, got {model!r}"
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
