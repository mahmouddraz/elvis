"""ELVIS: Electric Vehicle Charging Infrastructure Simulator.

ELVIS is the EV-charging application of the domain-agnostic :mod:`core` engine: vehicles
are core *requests*, their batteries *service profiles*, charging points *servers* and
stations / transformers capacity-limited *nodes*.

Typical use::

    from elvis import load_preset, simulate

    scenario = load_preset("office")
    result = simulate(scenario, "2024-01-01", "2024-12-31 23:45", "00:15:00", seed=42)
    print(result.summary())
"""

from importlib.metadata import PackageNotFoundError, version

from core.errors import ConfigurationError, CoreError, SimulationError
from core.requests import Request
from core.time import TimeGrid, TimeSeries
from elvis.demand import ChargingDemand
from elvis.infrastructure import (
    ChargingPointSpec,
    ChargingStationSpec,
    InfrastructureSpec,
    StorageSpec,
    TransformerSpec,
)
from elvis.presets import list_presets, load_preset
from elvis.realisation import Realisation
from elvis.result import ChargingResult
from elvis.scenario import OpeningHours, Operation, Scenario
from elvis.simulation import simulate
from elvis.vehicle import EVBattery, VehicleType

ChargingEvent = Request
"""A vehicle visit: a :class:`core.Request` whose level is the state of charge."""

ElvisError = CoreError

try:
    __version__ = version("elvislis")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "0.0.0"

__all__ = [
    "ChargingDemand",
    "ChargingEvent",
    "ChargingPointSpec",
    "ChargingResult",
    "ChargingStationSpec",
    "ConfigurationError",
    "EVBattery",
    "ElvisError",
    "InfrastructureSpec",
    "OpeningHours",
    "Operation",
    "Realisation",
    "Scenario",
    "SimulationError",
    "StorageSpec",
    "TimeGrid",
    "TimeSeries",
    "TransformerSpec",
    "VehicleType",
    "__version__",
    "list_presets",
    "load_preset",
    "simulate",
]
