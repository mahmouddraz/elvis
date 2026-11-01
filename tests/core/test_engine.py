import datetime as dt

import numpy as np
import pytest

from core import ConfigurationError, OpenHours, Operation, simulate
from core.engine import Simulation, WaitingQueue

from .conftest import MONDAY, make_request, make_system, realisation_of, star

ONE_SERVER = star(1)


def run(requests, operation=None, network=ONE_SERVER, hours=24):
    system = make_system(network, operation=operation)
    return Simulation(realisation_of(system, requests, hours=hours)).run()


def test_request_is_served_while_present_and_leaves_at_departure():
    result = run([make_request(1, 2, 3, level=0.0)])
    np.testing.assert_allclose(result.total_rate[:7], [0, 0, 11, 11, 11, 0, 0])
    (session,) = result.sessions
    assert (session.first_served_step, session.last_served_step) == (2, 4)


def test_second_request_is_rejected_without_queue():
    result = run([make_request(1, 0, 5), make_request(2, 1, 5)])
    assert result.rejections == 1
    assert result.rejection_rate == 0.5


def test_waiting_request_takes_over_the_freed_server():
    operation = Operation(queue_capacity=1)
    result = run([make_request(1, 0, 2, level=0.0), make_request(2, 1, 5, level=0.0)], operation)
    assert result.rejections == 0
    np.testing.assert_allclose(result.total_rate[:7], [11] * 6 + [0])
    assert [s.request_id for s in result.sessions] == [1, 2]


def test_waiting_request_leaves_after_its_dwell_time():
    operation = Operation(queue_capacity=1)
    result = run([make_request(1, 0, 5, level=0.0), make_request(2, 1, 2, level=0.0)], operation)
    assert [s.request_id for s in result.sessions] == [1]
    assert result.total_rate[5] == 0


def test_full_queue_rejects():
    operation = Operation(queue_capacity=1)
    result = run([make_request(i, 0, 5) for i in (1, 2, 3)], operation)
    assert result.rejections == 1


def test_release_at_target_frees_the_server_early():
    operation = Operation(queue_capacity=1, release="at_target")
    # capacity 50 at level 0.78 needs 11 units: one hour at rate 11
    result = run(
        [make_request(1, 0, 10, level=0.78), make_request(2, 0, 10, level=0.78)], operation
    )
    np.testing.assert_allclose(result.total_rate[:3], [11, 11, 0])
    assert result.rejections == 0


def test_closed_system_rejects_arrivals_and_releases_requests():
    operation = Operation(open_hours=OpenHours(8, 18), queue_capacity=2)
    requests = [make_request(1, 6, 2), make_request(2, 17, 5, level=0.0)]
    result = run(requests, operation)
    assert result.rejections == 1
    assert result.total_rate[17] == 11
    assert result.total_rate[18] == 11  # 18:00 is still open (inclusive)
    assert result.total_rate[19] == 0


def test_requests_off_the_grid_are_admitted_at_the_next_step():
    late = make_request(1, 1.5, 3, level=0.0)
    result = run([late])
    np.testing.assert_allclose(result.total_rate[:5], [0, 0, 11, 11, 11])


def test_free_servers_are_used_lowest_index_first():
    result = run(
        [make_request(1, 0, 3), make_request(2, 0, 3)],
        network=star(3),
    )
    assert result.rates[0, 2] == 0
    assert result.rates[0, :2].tolist() == [11, 11]


def test_progress_is_reported():
    seen = []
    system = make_system()
    Simulation(realisation_of(system, [], hours=48)).run(seen.append)
    assert seen[0] == 0 and seen[-1] == 1.0
    assert seen == sorted(seen)


def test_simulate_api():
    system = make_system(arrivals_per_week=200)
    result = simulate(system, "2024-01-01", "2024-01-07 23:45", seed=3)
    again = simulate(system, "2024-01-01", "2024-01-07 23:45", seed=3)
    np.testing.assert_array_equal(result.total_rate, again.total_rate)
    rerun = simulate(result.realisation)
    np.testing.assert_array_equal(result.total_rate, rerun.total_rate)
    with pytest.raises(ConfigurationError):
        simulate(system)
    with pytest.raises(ConfigurationError):
        simulate(result.realisation, "2024-01-01", "2024-01-02")


def test_waiting_queue():
    queue = WaitingQueue(2)
    first, second = make_request(1, 0, 3), make_request(2, 0, 1)
    queue.push(first)
    queue.push(second)
    assert queue.is_full and queue.next_departure == second.departure
    queue.drop_departed(MONDAY + dt.timedelta(hours=1))
    assert len(queue) == 1 and queue.next_departure == first.departure
    assert queue.pop() is first
    assert queue.next_departure == dt.datetime.max


def test_request_outcomes_account_for_every_unit_delivered():
    requests = [make_request(1, 0, 2, level=0.0), make_request(2, 0, 5), make_request(3, 1, 1)]
    result = run(requests, operation=Operation(queue_capacity=1))
    outcomes = result.request_outcomes()
    assert outcomes["connected"].tolist() == [True, True, False]
    np.testing.assert_allclose(outcomes["delivered"], [22, 25, 0])  # 2 waits, then takes over
    np.testing.assert_allclose(outcomes["demand"], [50, 25, 25])
    assert outcomes["delivered"].sum() == pytest.approx(result.total_quantity)
