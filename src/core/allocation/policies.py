"""The built-in allocation policies."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np

from core._numeric import ceil_to, floor_to
from core.allocation.base import Allocation, Policy, StepContext, allocate_in_priority_order
from core.errors import ConfigurationError
from core.network.runtime import Server
from core.requests import Request
from core.time.durations import format_duration, parse_duration


class Unconstrained(Policy):
    """Every request is served as fast as its profile and its server allow.

    Inner-node and root limits are ignored, which reveals the unmanaged peak. If a root's
    limit is exceeded, its buffer (if any) releases as much of the excess as it can.
    """

    name = "unconstrained"

    def allocate(self, context: StepContext) -> Allocation:
        allocation = Allocation.idle(context.network)
        for root in context.network.roots:
            total = 0.0
            for server in context.occupied:
                if server.root is not root:
                    continue
                session = server.session
                rate = min(
                    session.max_rate_at(session.level),
                    server.rate_to_reach(session.request.target_level, context.step_hours),
                    server.max_rate,
                )
                allocation.rates[server.index] = rate
                total += rate
            buffer = root.buffer
            if buffer is not None and total > root.max_rate:
                available = buffer.max_release_rate(0, context.step_hours)
                allocation.buffer_rates[buffer] = -min(ceil_to(total - root.max_rate), available)
        return allocation


class FirstComeFirstServed(Policy):
    """Respect all limits; requests that arrived first are served first."""

    name = "first_come_first_served"

    def allocate(self, context: StepContext) -> Allocation:
        allocation = Allocation.idle(context.network)
        by_arrival = sorted(
            context.occupied, key=lambda s: (s.session.request.arrival, s.session.request.id)
        )
        for root in context.network.roots:
            servers = [server for server in by_arrival if server.root is root]
            allocate_in_priority_order(servers, root, context, allocation)
        return allocation


class EarliestDepartureFirst(Policy):
    """Respect all limits; requests that leave soonest are served first.

    An earliest-deadline-first rule: it postpones requests with slack into later steps, which
    maximises the delivered quantity in many settings.

    Args:
        departure_noise_hours: Standard deviation of the error in the departure time the policy
            sees (e.g. a time announced by the user). The true departure is unaffected.
        seed: Seed of the announcement errors; each request gets a fixed error.
    """

    name = "earliest_departure_first"

    def __init__(self, departure_noise_hours: float = 0.0, seed: int = 0) -> None:
        if departure_noise_hours < 0:
            raise ConfigurationError("departure_noise_hours must be >= 0")
        self.departure_noise_hours = float(departure_noise_hours)
        self.seed = int(seed)
        self._announced: dict[int, dt.datetime] = {}

    def to_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {"policy": self.name}
        if self.departure_noise_hours:
            config |= {"departure_noise_hours": self.departure_noise_hours, "seed": self.seed}
        return config

    def announced_departure(self, request: Request) -> dt.datetime:
        """The departure time as the policy believes it."""
        if not self.departure_noise_hours:
            return request.departure
        announced = self._announced.get(request.id)
        if announced is None:
            error = np.random.default_rng((self.seed, request.id)).normal(
                0, self.departure_noise_hours
            )
            announced = request.departure + dt.timedelta(hours=float(error))
            self._announced[request.id] = announced
        return announced

    def allocate(self, context: StepContext) -> Allocation:
        allocation = Allocation.idle(context.network)
        by_departure = sorted(
            context.occupied, key=lambda s: self.announced_departure(s.session.request)
        )
        for root in context.network.roots:
            servers = [server for server in by_departure if server.root is root]
            allocate_in_priority_order(servers, root, context, allocation)
        return allocation


class FairShare(Policy):
    """Respect all limits and rotate scarce capacity fairly between requests.

    Capacity is handed out in blocks of ``block``: a request that has started a block keeps
    priority until the block is complete; otherwise requests served least so far go first.
    A step only counts as "served" when the network was not the limiting factor.
    """

    name = "fair_share"

    def __init__(self, block: str | dt.timedelta = dt.timedelta(minutes=15)) -> None:
        self.block = parse_duration(block, name="block")
        self._history: dict[Server, _ServiceHistory] = {}

    def to_config(self) -> dict[str, Any]:
        return {"policy": self.name, "block": format_duration(self.block)}

    def allocate(self, context: StepContext) -> Allocation:
        steps_per_block = int(max(self.block.total_seconds() / (context.step_hours * 3600), 1))
        self._update_history(context.occupied, steps_per_block)
        priority = self._priority_order(steps_per_block)
        allocation = Allocation.idle(context.network)

        def count_served(server: Server, rate: float, rate_to_target: float) -> None:
            limited_by_network = rate != rate_to_target and rate < server.max_rate_now()
            if not limited_by_network:
                self._history[server].steps_served += 1

        for root in context.network.roots:
            servers = [server for server in priority if server.root is root]
            allocate_in_priority_order(servers, root, context, allocation, count_served)
        return allocation

    def _update_history(self, occupied: list[Server], steps_per_block: int) -> None:
        history = self._history
        # Rebase counts so they do not grow without bound; whole blocks keep priorities intact.
        least = floor_to(min((h.steps_served for h in history.values()), default=0), 0)
        rebase = least - least % steps_per_block
        busy = set(occupied)
        for server, entry in list(history.items()):
            entry.steps_served -= rebase
            if server not in busy:
                del history[server]
            elif entry.request_id != server.session.request.id:
                history[server] = _ServiceHistory(server.session.request.id)
        for server in occupied:
            if server not in history:
                history[server] = _ServiceHistory(server.session.request.id)

    def _priority_order(self, steps_per_block: int) -> list[Server]:
        def blocks(server: Server) -> float:
            return self._history[server].steps_served / steps_per_block

        order = sorted(self._history, key=blocks)
        order.sort(key=lambda server: round(blocks(server) % 1, 2), reverse=True)
        return order


@dataclass(slots=True)
class _ServiceHistory:
    """Steps a request has been served without network limitation (rebased to blocks)."""

    request_id: int
    steps_served: float = 0
