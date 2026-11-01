"""Stateful network used during one simulation run.

Rate assignments are passed around as a plain list indexed by :attr:`Server.index`; one
flat list (instead of dictionaries keyed by node objects) keeps per-step bookkeeping cheap and
the iteration order deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from core._numeric import floor_to
from core.errors import SimulationError
from core.network.spec import BufferSpec, NetworkSpec, NodeSpec, ServerSpec
from core.requests import Request


@dataclass(slots=True)
class Session:
    """A request being served and its current fill level.

    The profile's capacity and rate limit are cached here because they are read for every
    occupied server in every step.
    """

    request: Request
    level: float
    capacity: float = field(init=False)
    max_rate_at: Callable[[float], float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.capacity = self.request.profile.capacity
        self.max_rate_at = self.request.profile.max_rate_at


class Server:
    """Serves at most one request at a time."""

    __slots__ = ("ancestors", "id", "index", "max_rate", "min_rate", "parent", "root", "session")

    def __init__(self, spec: ServerSpec, id: str, index: int, parent: Node) -> None:
        self.id = id
        self.index = index
        self.max_rate = spec.max_rate
        self.min_rate = spec.min_rate
        self.parent = parent
        self.ancestors: tuple[Node, ...] = (parent, *parent.ancestors)
        self.root: Node = self.ancestors[-1]
        self.session: Session | None = None

    def __repr__(self) -> str:
        return f"Server({self.id!r})"

    def connect(self, request: Request) -> None:
        self.session = Session(request, request.initial_level)

    def disconnect(self) -> Session:
        session = self.session
        assert session is not None, f"{self.id} is idle"
        self.session = None
        return session

    def max_rate_now(self) -> float:
        """Highest rate the server and the current request's profile accept."""
        session = self.session
        if session is None:
            return 0.0
        return floor_to(min(self.max_rate, session.max_rate_at(session.level)))

    def rate_to_reach(self, level: float, step_hours: float) -> float:
        """Average rate that brings the current request to ``level`` within one step."""
        session = self.session
        assert session is not None
        if session.level > level:
            return 0.0
        return (level - session.level) * session.capacity / step_hours

    def serve(self, rate: float, step_hours: float) -> None:
        """Apply ``rate`` for one step; the level saturates at 1."""
        session = self.session
        assert session is not None
        session.level = min(1, session.level + rate * step_hours / session.capacity)


class Node:
    """An inner node of the network tree."""

    __slots__ = ("ancestors", "buffer", "children", "id", "max_rate", "min_rate", "servers")

    def __init__(self, spec: NodeSpec, id: str, parent: Node | None) -> None:
        self.id = id
        self.max_rate = spec.max_rate
        self.min_rate = spec.min_rate
        self.ancestors: tuple[Node, ...] = () if parent is None else (parent, *parent.ancestors)
        self.children: list[Node | Server] = []
        self.servers: list[Server] = []
        self.buffer: Buffer | None = None

    def __repr__(self) -> str:
        return f"Node({self.id!r})"

    def headroom(self, rates: Sequence[float], background: float = 0.0) -> float:
        """Rate still available after ``background`` and the assignments in ``rates``."""
        assigned = background
        for server in self.servers:
            assigned += rates[server.index]
        return floor_to(max(self.max_rate - assigned, 0))


class Buffer:
    """A store attached to a root node."""

    __slots__ = ("id", "level", "node", "spec")

    def __init__(self, spec: BufferSpec, id: str, node: Node) -> None:
        self.id = id
        self.spec = spec
        self.node = node
        self.level = spec.initial_level

    def __repr__(self) -> str:
        return f"Buffer({self.id!r}, level={self.level:.3f})"

    @property
    def _one_way_efficiency(self) -> float:
        """Square root of the round-trip efficiency, applied when filling and when draining."""
        return math.sqrt(self.spec.efficiency)

    def max_release_rate(self, already_released: float, step_hours: float) -> float:
        """Highest additional rate the buffer can deliver given what it already delivers.

        Delivering ``rate`` for one step drains ``rate * step_hours / eta`` from storage, where
        ``eta`` is the one-way efficiency.
        """
        capacity = self.spec.capacity
        eta = self._one_way_efficiency
        rate_limit = self.spec.max_rate - already_released
        level_after = already_released * step_hours / eta / capacity
        rate_to_empty = (
            (self.level - level_after - self.spec.min_level) * capacity * eta / step_hours
        )
        return floor_to(max(min(rate_limit, rate_to_empty), 0))

    def absorb(self, available: float, step_hours: float) -> float:
        """Fill with up to ``available`` (drawn from the grid); return the rate drawn.

        Of ``rate * step_hours`` drawn, ``eta`` times that reaches storage.
        """
        capacity = self.spec.capacity
        eta = self._one_way_efficiency
        rate_to_full = (1 - self.level) * capacity / eta / step_hours
        rate = min(rate_to_full, self.spec.max_rate, available)
        self.level += rate * step_hours * eta / capacity
        return rate

    def release(self, rate: float, step_hours: float) -> None:
        """Deliver exactly ``rate``, which must be feasible."""
        if rate > self.max_release_rate(0, step_hours):
            raise SimulationError(f"{self.id}: cannot release {rate} at level {self.level}")
        self.level -= rate * step_hours / self._one_way_efficiency / self.spec.capacity


class Network:
    """The runtime network built from a :class:`NetworkSpec`."""

    def __init__(self, spec: NetworkSpec) -> None:
        self.spec = spec
        self.roots: list[Node] = []
        self.nodes: list[Node] = []
        self.servers: list[Server] = []
        for root_spec in spec.roots:
            root = self._build(root_spec, None)
            if root_spec.buffer is not None:
                buffer_id = root_spec.buffer.id or f"buffer-{len(self.roots) + 1}"
                root.buffer = Buffer(root_spec.buffer, buffer_id, root)
            self.roots.append(root)

    def _build(self, spec: NodeSpec, parent: Node | None) -> Node:
        node = Node(spec, spec.id or f"node-{len(self.nodes) + 1}", parent)
        self.nodes.append(node)
        for child in spec.children:
            if isinstance(child, NodeSpec):
                child_node = self._build(child, node)
                node.children.append(child_node)
                node.servers += child_node.servers
            else:
                index = len(self.servers)
                server = Server(child, child.id or f"server-{index + 1}", index, node)
                self.servers.append(server)
                node.children.append(server)
                node.servers.append(server)
        return node

    @property
    def buffers(self) -> list[Buffer]:
        return [root.buffer for root in self.roots if root.buffer is not None]
