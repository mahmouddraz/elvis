"""Invariants that must hold for any network, demand, profile and limit-aware policy."""

from dataclasses import dataclass

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from core import (
    BufferSpec,
    Demand,
    NetworkSpec,
    NodeSpec,
    Operation,
    ServerSpec,
    System,
    TimeSeries,
    TruncatedNormal,
    WeeklyProfileArrivals,
    simulate,
)

TOLERANCE = 1e-6


@dataclass(frozen=True)
class TaperingTank:
    """Accepts ``max_rate`` until ``knee``, then linearly less down to 10 % at level 1."""

    name: str
    capacity: float
    max_rate: float
    knee: float

    def max_rate_at(self, level: float) -> float:
        if level <= self.knee:
            return self.max_rate
        return self.max_rate * (1 - 0.9 * (level - self.knee) / (1 - self.knee))


@st.composite
def trees(draw, depth):
    rate = draw(st.sampled_from([5.0, 11.0, 22.0]))
    if depth == 0:
        return ServerSpec(rate)
    children = tuple(draw(trees(depth - 1)) for _ in range(draw(st.integers(1, 3))))
    capacity = sum(c.max_rate for c in children) * draw(st.sampled_from([0.4, 0.8, 1.0]))
    return NodeSpec(max(capacity, 1.0), children)


@st.composite
def systems(draw):
    root = draw(trees(draw(st.integers(1, 3))))
    if isinstance(root, ServerSpec):  # pragma: no cover - depth >= 1
        root = NodeSpec(root.max_rate, (root,))
    buffer = draw(
        st.none()
        | st.builds(
            BufferSpec,
            capacity=st.sampled_from([10.0, 40.0]),
            max_rate=st.sampled_from([5.0, 15.0]),
            initial_level=st.sampled_from([0.0, 0.5, 1.0]),
        )
    )
    root = NodeSpec(root.max_rate, root.children, buffer=buffer)
    profiles = (
        TaperingTank("small", 20, 7, 0.6),
        TaperingTank("large", 80, 30, draw(st.sampled_from([0.5, 0.9]))),
    )
    demand = Demand(
        WeeklyProfileArrivals(
            draw(st.integers(20, 400)), (1.0,) * 24, TruncatedNormal(3, 2, 0, 12)
        ),
        TruncatedNormal(0.3, 0.2, 0, 1),
        profiles,
        (0.7, 0.3),
    )
    operation = Operation(
        policy=draw(st.sampled_from(["earliest_departure_first", "fair_share"])),
        queue_capacity=draw(st.integers(0, 3)),
        release=draw(st.sampled_from(["at_departure", "at_target"])),
    )
    background = draw(st.sampled_from([None, 0.2]))
    return System(
        NetworkSpec((root,)),
        demand,
        operation,
        None if background is None else TimeSeries.constant(background * root.max_rate),
    )


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(system=systems(), seed=st.integers(0, 10_000))
def test_rates_respect_every_limit(system, seed):
    result = simulate(system, "2024-01-01", "2024-01-03 23:45", "15min", seed=seed)
    rates = result.rates
    servers = system.network.servers
    assert np.all(rates >= -TOLERANCE)
    assert np.all(rates <= np.array([s.max_rate for s in servers]) + TOLERANCE)

    def check(node, offset):
        # A root with a buffer may serve more than its own limit: the buffer supplies the
        # difference. Its net load is checked separately below.
        if node.buffer is None:
            width = len(node.servers)
            served = rates[:, offset : offset + width].sum(axis=1)
            assert np.all(served <= node.max_rate + TOLERANCE)
        for child in node.children:
            if isinstance(child, NodeSpec):
                check(child, offset)
            offset += len(child.servers) if isinstance(child, NodeSpec) else 1

    (root,) = system.network.roots
    check(root, 0)
    assert np.all(result.root_load <= root.max_rate + TOLERANCE)

    if root.buffer is not None:
        spec = root.buffer
        stored = spec.initial_level * spec.capacity + np.cumsum(result.total_buffer_rate) * 0.25
        assert np.all(stored >= spec.min_level * spec.capacity - TOLERANCE)
        assert np.all(stored <= spec.capacity + TOLERANCE)


@settings(max_examples=25, deadline=None)
@given(system=systems(), seed=st.integers(0, 10_000))
def test_delivered_quantity_never_exceeds_demand(system, seed):
    result = simulate(system, "2024-01-01", "2024-01-02 23:45", "15min", seed=seed)
    demanded = sum(request.demand for request in result.realisation.requests)
    assert result.total_quantity <= demanded + TOLERANCE
    assert result.rejections <= result.num_requests


@settings(max_examples=25, deadline=None)
@given(system=systems(), seed=st.integers(0, 10_000))
def test_profile_rate_limit_holds(system, seed):
    """No request is ever served faster than its profile allows at its current level."""
    result = simulate(system, "2024-01-01", "2024-01-01 23:45", "15min", seed=seed)
    # Replay: the level after each step follows from the recorded rates.
    by_server = {}
    for record in result.sessions:
        by_server.setdefault(record.server_id, []).append(record)
    requests = {r.id: r for r in result.realisation.requests}
    for column, server_id in enumerate(result.server_ids):
        for record in by_server.get(server_id, []):
            request = requests[record.request_id]
            level = request.initial_level
            for step in range(record.first_served_step, record.last_served_step + 1):
                rate = result.rates[step, column]
                assert rate <= request.profile.max_rate_at(level) + TOLERANCE
                level = min(1, level + rate * 0.25 / request.profile.capacity)
