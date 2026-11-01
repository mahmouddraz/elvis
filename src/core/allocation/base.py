"""Allocation interface and the routine shared by limit-aware policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from core._numeric import ceil_to, floor_to
from core.network.runtime import Buffer, Network, Node, Server


@dataclass(frozen=True, slots=True)
class StepContext:
    """What a policy may look at when allocating rates for one time step.

    Attributes:
        network: The network, including the sessions on occupied servers.
        occupied: Occupied servers, in the order their requests were connected.
        step: Index of the current time step.
        step_hours: Length of a time step in hours.
        background: Load on each root node that is not caused by requests.
    """

    network: Network
    occupied: list[Server]
    step: int
    step_hours: float
    background: float


@dataclass(slots=True)
class Allocation:
    """Rates decided for one time step.

    Attributes:
        rates: Rate per server, indexed by :attr:`Server.index`.
        buffer_rates: Rate per buffer. Negative values release from the buffer to supply
            servers; zero lets the buffer absorb leftover root capacity.
    """

    rates: list[float]
    buffer_rates: dict[Buffer, float] = field(default_factory=dict)

    @classmethod
    def idle(cls, network: Network) -> Allocation:
        return cls([0.0] * len(network.servers), dict.fromkeys(network.buffers, 0.0))


class Policy(ABC):
    """Decides the service rate of every occupied server in a time step.

    A new instance is created for every simulation run, so policies may keep state.
    """

    name: ClassVar[str]

    @abstractmethod
    def allocate(self, context: StepContext) -> Allocation:
        """Return the allocation for the current step."""

    def to_config(self) -> dict[str, Any]:
        return {"policy": self.name}

    def __str__(self) -> str:
        return self.name


def allocate_in_priority_order(
    servers: Iterable[Server],
    root: Node,
    context: StepContext,
    allocation: Allocation,
    on_assigned: Callable[[Server, float, float], None] | None = None,
) -> None:
    """Give each server, in order, as much rate as every limit above it still allows.

    Limits are the request's profile, the server, the remaining capacity of every inner node
    above it, and the root's remaining capacity (after background load) plus whatever its
    buffer can still release. Release needed beyond the root's own capacity is recorded in
    ``allocation.buffer_rates``.

    Args:
        servers: Occupied servers below ``root``, highest priority first.
        root: The root of their tree.
        context: The current step.
        allocation: Updated in place.
        on_assigned: Called with ``(server, rate, rate_to_target)`` after each assignment.
    """
    rates = allocation.rates
    buffer = root.buffer
    root_capacity = root.headroom(rates, context.background)
    total = 0.0
    release = 0.0
    buffer_available = 0.0

    for server in servers:
        available = root.headroom(rates, context.background)
        if buffer is not None:
            buffer_available = buffer.max_release_rate(release, context.step_hours)
            available += buffer_available
        limit = min(server.max_rate_now(), floor_to(available))
        for node in server.ancestors[:-1]:
            limit = min(limit, node.headroom(rates))
        rate_to_target = floor_to(
            server.rate_to_reach(server.session.request.target_level, context.step_hours)
        )
        rate = min(rate_to_target, limit)
        total += rate
        if total > root_capacity and buffer_available != 0:
            # Round up so the root never carries the rounding remainder; the cap keeps the
            # release feasible.
            release = min(
                ceil_to(total - root_capacity), buffer.max_release_rate(0, context.step_hours)
            )
        rates[server.index] = rate
        if on_assigned is not None:
            on_assigned(server, rate, rate_to_target)

    if buffer is not None:
        allocation.buffer_rates[buffer] = -release
