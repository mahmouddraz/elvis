"""core: a domain-agnostic engine for stochastic service systems.

Requests arrive at random (:mod:`core.arrivals`), occupy servers in a capacity-limited
network (:mod:`core.network`), wait in a bounded queue if all servers are busy, and are served
at rates chosen by an allocation policy (:mod:`core.allocation`) until they reach a target
level or their dwell time ends. The engine (:mod:`core.engine`) steps through a time grid and
records every rate.

Applications supply the domain vocabulary and physics; for instance :mod:`elvis` maps
electric vehicles to requests, batteries to service profiles and charging sites to networks.

Minimal example with a custom profile::

    from dataclasses import dataclass
    import core

    @dataclass(frozen=True)
    class Tank:                       # any object with name, capacity, max_rate_at()
        name: str = "tank"
        capacity: float = 10.0
        def max_rate_at(self, level): return 5.0

    system = core.System(
        network=core.NetworkSpec((core.NodeSpec(8.0, (core.ServerSpec(5.0),) * 2),)),
        demand=core.Demand(
            arrivals=core.WeeklyProfileArrivals(50, (1.0,) * 168, core.TruncatedNormal(2, 1, 0, 6)),
            initial_level=core.TruncatedNormal(0.3, 0.1, 0, 1),
            profiles=(Tank(),),
            weights=(1.0,),
        ),
        operation=core.Operation(policy="earliest_departure_first"),
    )
    result = core.simulate(system, "2024-01-01", "2024-01-07", seed=1)
"""

from core.allocation import (
    EarliestDepartureFirst,
    FairShare,
    FirstComeFirstServed,
    Policy,
    Unconstrained,
    register_policy,
)
from core.arrivals import GaussianMixtureArrivals, WeeklyProfileArrivals, register_arrival_process
from core.demand import Demand, DemandModel, EmpiricalDemand
from core.engine import Result, Simulation, simulate
from core.errors import ConfigurationError, CoreError, SimulationError
from core.network import BufferSpec, NetworkSpec, NodeSpec, ServerSpec
from core.random import TruncatedNormal
from core.requests import Request, ServiceProfile
from core.system import OpenHours, Operation, Realisation, System
from core.time import TimeGrid, TimeSeries

__all__ = [
    "BufferSpec",
    "ConfigurationError",
    "CoreError",
    "Demand",
    "DemandModel",
    "EarliestDepartureFirst",
    "EmpiricalDemand",
    "FairShare",
    "FirstComeFirstServed",
    "GaussianMixtureArrivals",
    "NetworkSpec",
    "NodeSpec",
    "OpenHours",
    "Operation",
    "Policy",
    "Realisation",
    "Request",
    "Result",
    "ServerSpec",
    "ServiceProfile",
    "Simulation",
    "SimulationError",
    "System",
    "TimeGrid",
    "TimeSeries",
    "TruncatedNormal",
    "Unconstrained",
    "WeeklyProfileArrivals",
    "register_arrival_process",
    "register_policy",
    "simulate",
]
