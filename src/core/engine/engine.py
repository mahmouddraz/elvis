"""The time-stepped simulation of one realisation.

Each time step runs, in order:

1. release: waiting and served requests whose time is up leave (all leave when the system
   closes); freed servers immediately take the next waiting request;
2. admit: new arrivals take a free server, join the queue, or are rejected;
3. allocate: the policy assigns rates to occupied servers and buffers;
4. serve: request levels and buffer levels advance by one step;
5. record: rates are written to the result.

Subclasses can override the phase methods to model domain-specific behaviour.
"""

from __future__ import annotations

import datetime as dt
import heapq
import logging
from collections.abc import Callable

import numpy as np

from core.allocation import Allocation, StepContext
from core.engine.queue import WaitingQueue
from core.engine.result import Result, SessionRecord
from core.network.runtime import Network, Server
from core.requests import Request
from core.system import Realisation

logger = logging.getLogger(__name__)

LEVEL_TOLERANCE_DECIMALS = 3
"""A request counts as complete once its level rounded to this many decimals hits the target."""

ProgressCallback = Callable[[float], None]


class Simulation:
    """Runs one realisation once. Create a new instance for every run."""

    def __init__(self, realisation: Realisation) -> None:
        self.realisation = realisation
        system = realisation.system
        self.grid = realisation.grid
        self.network = Network(system.network)
        self.policy = system.operation.create_policy()
        self.open_hours = system.operation.open_hours
        self.release_at_target = system.operation.release == "at_target"
        self.queue = WaitingQueue(system.operation.queue_capacity)
        self.background = realisation.background()

        self._free: list[int] = list(range(len(self.network.servers)))
        self._occupied: list[Server] = []
        self._next_request = 0
        self._rejections = 0
        self._sessions: dict[int, SessionRecord] = {}
        self._delivered: dict[int, float] = {}
        self._rates = np.zeros((self.grid.num_steps, len(self.network.servers)))
        self._buffer_rates = np.zeros((self.grid.num_steps, len(self.network.buffers)))

    def run(self, progress: ProgressCallback | None = None) -> Result:
        report_every = max(1, self.grid.num_steps // 20)
        for step, now in enumerate(self.grid.timestamps):
            is_open = self.open_hours is None or self.open_hours.contains(now)
            self.release_waiting(now, is_open)
            self.release_served(now, is_open)
            self.admit_arrivals(now, is_open)
            allocation = self.policy.allocate(
                StepContext(
                    self.network,
                    self._occupied,
                    step,
                    self.grid.step_hours,
                    float(self.background[step]),
                )
            )
            self.track_sessions(step, allocation)
            self.serve(allocation)
            self.operate_buffers(step, allocation)
            self._rates[step] = allocation.rates
            if progress is not None and step % report_every == 0:
                progress(step / self.grid.num_steps)
        if progress is not None:
            progress(1.0)
        return Result(
            self.realisation,
            [server.id for server in self.network.servers],
            [buffer.id for buffer in self.network.buffers],
            self._rates,
            self._buffer_rates,
            self._rejections,
            list(self._sessions.values()),
            self._delivered,
        )

    # -- step phases -----------------------------------------------------------------------

    def release_waiting(self, now: dt.datetime, is_open: bool) -> None:
        if not is_open:
            self.queue.clear()
        elif not self.release_at_target:
            self.queue.drop_departed(now)

    def release_served(self, now: dt.datetime, is_open: bool) -> None:
        still_occupied = []
        for server in self._occupied:
            if is_open and not self.is_done(server, now):
                still_occupied.append(server)
                continue
            server.disconnect()
            logger.debug("%s: %s released", now, server.id)
            if is_open and len(self.queue):
                self.connect(server, self.queue.pop())
                still_occupied.append(server)
            else:
                heapq.heappush(self._free, server.index)
        self._occupied = still_occupied

    def is_done(self, server: Server, now: dt.datetime) -> bool:
        session = server.session
        if self.release_at_target:
            return round(session.level, LEVEL_TOLERANCE_DECIMALS) >= session.request.target_level
        return session.request.departure <= now

    def admit_arrivals(self, now: dt.datetime, is_open: bool) -> None:
        requests = self.realisation.requests
        while self._next_request < len(requests) and requests[self._next_request].arrival <= now:
            request = requests[self._next_request]
            self._next_request += 1
            if is_open and self._free:
                server = self.network.servers[heapq.heappop(self._free)]
                self.connect(server, request)
                self._occupied.append(server)
            elif is_open and not self.queue.is_full:
                self.queue.push(request)
            else:
                self._rejections += 1
                logger.debug("%s: request %d rejected", now, request.id)

    def connect(self, server: Server, request: Request) -> None:
        server.connect(request)
        self._delivered[request.id] = 0.0

    def track_sessions(self, step: int, allocation: Allocation) -> None:
        for server in self._occupied:
            if allocation.rates[server.index] > 0:
                request_id = server.session.request.id
                record = self._sessions.get(request_id)
                if record is None:
                    self._sessions[request_id] = SessionRecord(request_id, server.id, step, step)
                else:
                    record.last_served_step = step

    def serve(self, allocation: Allocation) -> None:
        step_hours = self.grid.step_hours
        for server in self._occupied:
            rate = allocation.rates[server.index]
            server.serve(rate, step_hours)
            self._delivered[server.session.request.id] += rate * step_hours

    def operate_buffers(self, step: int, allocation: Allocation) -> None:
        for column, buffer in enumerate(self.network.buffers):
            rate = allocation.buffer_rates.get(buffer, 0.0)
            if rate == 0:
                headroom = buffer.node.headroom(allocation.rates, float(self.background[step]))
                rate = buffer.absorb(headroom, self.grid.step_hours)
            else:
                buffer.release(-rate, self.grid.step_hours)
            self._buffer_rates[step, column] = rate
