"""Declarative description of a capacity network.

A network is a forest of trees. Leaves are :class:`ServerSpec` (each serves one request at a
time); inner nodes are :class:`NodeSpec` whose rate limit is shared by everything below
them. A root node may hold a :class:`BufferSpec`, a store that can supply extra rate when the
root's own limit is reached and absorbs spare capacity otherwise.

Example (an electrical site): root = grid connection, inner nodes = cabinets or stations,
servers = connectors, buffer = stationary battery.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.config.validation import ProblemCollector
from core.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ServerSpec:
    """A leaf that serves at most one request at a time.

    Attributes:
        max_rate: Highest service rate.
        min_rate: Lowest non-zero service rate (informational; not enforced yet).
        id: Optional identifier; generated if omitted.
    """

    max_rate: float
    min_rate: float = 0.0
    id: str | None = None

    def __post_init__(self) -> None:
        check_rate_limits("server", self.min_rate, self.max_rate)


@dataclass(frozen=True, slots=True)
class BufferSpec:
    """A store attached to a root node.

    Attributes:
        capacity: Quantity it holds when full.
        max_rate: Highest rate in either direction.
        min_rate: Lowest non-zero rate (informational).
        efficiency: Round-trip efficiency in (0, 1]; half of the loss (as a square root)
            applies when filling, half when draining.
        initial_level: Fill level at the start, in [0, 1].
        min_level: The buffer is never drained below this level.
        id: Optional identifier; generated if omitted.
    """

    capacity: float
    max_rate: float
    min_rate: float = 0.0
    efficiency: float = 1.0
    initial_level: float = 0.0
    min_level: float = 0.0
    id: str | None = None

    def __post_init__(self) -> None:
        problems = ProblemCollector("buffer")
        problems.positive("capacity", self.capacity)
        problems.fraction("efficiency", self.efficiency)
        if self.efficiency == 0:
            problems.add("efficiency must be > 0")
        problems.fraction("initial_level", self.initial_level)
        problems.fraction("min_level", self.min_level)
        if self.min_level > self.initial_level:
            problems.add("min_level exceeds initial_level")
        problems.raise_if_any()
        check_rate_limits("buffer", self.min_rate, self.max_rate)


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """An inner node whose rate limit is shared by all servers below it.

    Attributes:
        max_rate: Highest total rate through the node (for roots including background load).
        children: Nodes or servers below.
        min_rate: Lowest non-zero rate (informational).
        buffer: Optional store; only allowed on root nodes.
        id: Optional identifier; generated if omitted.
    """

    max_rate: float
    children: tuple[NodeSpec | ServerSpec, ...]
    min_rate: float = 0.0
    buffer: BufferSpec | None = None
    id: str | None = None

    def __post_init__(self) -> None:
        check_rate_limits("node", self.min_rate, self.max_rate)
        if not self.children:
            raise ConfigurationError("node: needs at least one child")
        if any(isinstance(c, NodeSpec) and c.buffer is not None for c in self.children):
            raise ConfigurationError("buffers are only supported on root nodes")

    @property
    def servers(self) -> list[ServerSpec]:
        """All servers below this node, depth first."""
        found: list[ServerSpec] = []
        for child in self.children:
            found += child.servers if isinstance(child, NodeSpec) else [child]
        return found


@dataclass(frozen=True, slots=True)
class NetworkSpec:
    """All trees of a system."""

    roots: tuple[NodeSpec, ...]

    def __post_init__(self) -> None:
        if not self.roots:
            raise ConfigurationError("network: needs at least one root node")

    @property
    def servers(self) -> list[ServerSpec]:
        """All servers, depth first."""
        return [server for root in self.roots for server in root.servers]

    @property
    def nominal_rate(self) -> float:
        """Sum of the maximum rate of all servers."""
        return sum(server.max_rate for server in self.servers)


def check_rate_limits(context: str, min_rate: float, max_rate: float) -> None:
    """Validate a ``(min, max)`` rate pair."""
    problems = ProblemCollector(context)
    problems.positive("max rate", max_rate)
    problems.non_negative("min rate", min_rate)
    if not problems.problems and min_rate > max_rate:
        problems.add("min rate exceeds max rate")
    problems.raise_if_any()
