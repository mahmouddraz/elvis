"""First-in-first-out queue of requests waiting for a free server."""

from __future__ import annotations

import datetime as dt
from collections import deque

from core.requests import Request

_NEVER = dt.datetime.max


class WaitingQueue:
    """Bounded FIFO queue that knows when its next request gives up and leaves."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._requests: deque[Request] = deque()
        self.next_departure = _NEVER

    def __len__(self) -> int:
        return len(self._requests)

    @property
    def is_full(self) -> bool:
        return len(self._requests) >= self.capacity

    def push(self, request: Request) -> None:
        self._requests.append(request)
        self.next_departure = min(self.next_departure, request.departure)

    def pop(self) -> Request:
        request = self._requests.popleft()
        self._refresh_next_departure()
        return request

    def clear(self) -> None:
        self._requests.clear()
        self.next_departure = _NEVER

    def drop_departed(self, now: dt.datetime) -> None:
        """Remove requests whose dwell time ran out while they were waiting."""
        if now < self.next_departure:
            return
        self._requests = deque(r for r in self._requests if r.departure > now)
        self._refresh_next_departure()

    def _refresh_next_departure(self) -> None:
        self.next_departure = min((r.departure for r in self._requests), default=_NEVER)
