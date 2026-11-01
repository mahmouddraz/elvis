import numpy as np
import pytest

from core import (
    BufferSpec,
    ConfigurationError,
    NetworkSpec,
    NodeSpec,
    Operation,
    ServerSpec,
    TimeSeries,
)
from core.allocation import (
    POLICIES,
    Allocation,
    EarliestDepartureFirst,
    FairShare,
    Policy,
    policy_from_config,
    register_policy,
)
from core.engine import Simulation

from .conftest import Tank, make_request, make_system, realisation_of, star

TIGHT = star(3, root_rate=15)


def run(policy, requests, network=TIGHT, **system_changes):
    system = make_system(network, operation=Operation(policy=policy), **system_changes)
    return Simulation(realisation_of(system, requests)).run()


def three_long_stays():
    big = Tank(capacity=500)
    return [make_request(i, 0, 10 + i, level=0.1, profile=big) for i in (1, 2, 3)]


def test_unconstrained_ignores_the_root_limit():
    assert run("unconstrained", three_long_stays()).total_rate[0] == 33


def test_unconstrained_stops_at_the_target_level():
    result = run("unconstrained", [make_request(1, 0, 10, level=0.9)])  # needs 5 units
    np.testing.assert_allclose(result.total_rate[:3], [5, 0, 0])


def test_earliest_departure_first_order():
    first_step = run("earliest_departure_first", three_long_stays()).rates[0]
    np.testing.assert_allclose(first_step, [11, 4, 0])  # request 1 leaves first


def test_arrival_order_vs_departure_order():
    big = Tank(capacity=500)
    early_long = make_request(1, 0, 20, level=0.1, profile=big)
    late_short = make_request(2, 1, 2, level=0.1, profile=big)
    network = star(2, root_rate=11)
    fcfs = run("first_come_first_served", [early_long, late_short], network=network)
    edf = run("earliest_departure_first", [early_long, late_short], network=network)
    np.testing.assert_allclose(fcfs.rates[1], [11, 0])  # the earlier arrival keeps priority
    np.testing.assert_allclose(edf.rates[1], [0, 11])  # the earlier departure takes over


def test_inner_node_limits_are_respected():
    network = star(2, servers_per_node=2, node_rate=12)
    result = run("earliest_departure_first", three_long_stays()[:2], network=network)
    np.testing.assert_allclose(result.rates[0], [11, 1])


def test_limits_of_deeper_trees_are_respected():
    leaf = NodeSpec(20, (ServerSpec(11), ServerSpec(11)))
    middle = NodeSpec(25, (leaf, leaf))
    network = NetworkSpec((NodeSpec(100, (middle,)),))
    requests = [make_request(i, 0, 10, level=0.1, profile=Tank(capacity=500)) for i in range(1, 5)]
    rates = run("earliest_departure_first", requests, network=network).rates[0]
    assert rates[:2].sum() <= 20
    assert rates.sum() == 25


def test_background_load_reduces_root_capacity():
    result = run("earliest_departure_first", three_long_stays(), background=TimeSeries.constant(10))
    assert result.total_rate[0] == 5


def test_fair_share_rotates_capacity():
    result = run({"policy": "fair_share", "block": "01:00:00"}, three_long_stays())
    served = result.rates[:6] > 0
    # 15 units/h serve one request fully and a second partially; full service rotates hourly.
    assert served.sum(axis=0).min() >= 2
    assert np.all(result.total_rate[:6] == 15)


def test_buffer_covers_peaks_until_empty():
    network = star(3, root_rate=15, buffer=BufferSpec(20, 10, initial_level=1.0))
    result = run("earliest_departure_first", three_long_stays(), network=network)
    assert result.total_rate[0] == 25
    assert result.total_buffer_rate[0] == -10
    assert result.total_rate[2] == 15


def test_empty_buffer_absorbs_spare_root_capacity():
    network = star(3, root_rate=15, buffer=BufferSpec(20, 10))
    result = run("earliest_departure_first", [make_request(1, 0, 10, level=0.1)], network=network)
    np.testing.assert_allclose(result.total_rate[:6], [11, 11, 11, 11, 1, 0], atol=0.01)
    np.testing.assert_allclose(result.total_buffer_rate[:6], [4, 4, 4, 4, 4, 0])


def test_unconstrained_releases_buffer_for_excess():
    network = star(3, root_rate=15, buffer=BufferSpec(40, 10, initial_level=1.0))
    assert run("unconstrained", three_long_stays(), network=network).total_buffer_rate[0] == -10


def test_policy_registry():
    assert set(POLICIES) == {
        "unconstrained",
        "first_come_first_served",
        "earliest_departure_first",
        "fair_share",
    }
    assert isinstance(policy_from_config("earliest_departure_first"), EarliestDepartureFirst)
    policy = policy_from_config({"policy": "fair_share", "block": "00:30:00"})
    assert isinstance(policy, FairShare)
    assert policy.to_config() == {"policy": "fair_share", "block": "00:30:00"}
    with pytest.raises(ConfigurationError, match="must be one of"):
        policy_from_config("fcfs")
    with pytest.raises(ConfigurationError, match="invalid parameters"):
        policy_from_config({"policy": "fair_share", "speed": 1})


def test_custom_policy_can_be_registered():
    @register_policy
    class Idle(Policy):
        name = "idle"

        def allocate(self, context):
            return Allocation.idle(context.network)

    try:
        assert run("idle", three_long_stays()).total_quantity == 0
    finally:
        del POLICIES["idle"]


def test_noisy_departures_degrade_edf_gracefully():
    big = Tank(capacity=500)
    early_long = make_request(1, 0, 20, level=0.1, profile=big)
    late_short = make_request(2, 1, 2, level=0.1, profile=big)
    network = star(2, root_rate=11)
    exact = run({"policy": "earliest_departure_first"}, [early_long, late_short], network=network)
    noisy = run(
        {"policy": "earliest_departure_first", "departure_noise_hours": 1000, "seed": 1},
        [early_long, late_short],
        network=network,
    )
    np.testing.assert_allclose(exact.rates[1], [0, 11])
    assert noisy.rates[1].sum() == 11  # still a valid allocation, possibly in the wrong order
    policy = policy_from_config({"policy": "earliest_departure_first", "departure_noise_hours": 2})
    assert policy.announced_departure(early_long) == policy.announced_departure(early_long)
    assert policy.announced_departure(early_long) != early_long.departure
