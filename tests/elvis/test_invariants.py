"""EV-level invariants: the EV vocabulary maps onto core without losing physical limits."""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from elvis import InfrastructureSpec, Operation, StorageSpec, TimeGrid, TimeSeries, simulate

from .conftest import make_scenario

TOLERANCE = 1e-6


@st.composite
def scenarios(draw):
    points_per_station = draw(st.integers(1, 3))
    stations = draw(st.integers(1, 4))
    point_kw = draw(st.sampled_from([3.7, 11.0, 22.0, 50.0]))
    station_kw = point_kw * points_per_station * draw(st.sampled_from([0.5, 1.0]))
    transformer_kw = (
        point_kw * points_per_station * stations * draw(st.sampled_from([0.3, 0.7, 1.0]))
    )
    storage = draw(
        st.none()
        | st.builds(
            StorageSpec,
            capacity_kwh=st.sampled_from([10.0, 50.0]),
            max_power_kw=st.sampled_from([5.0, 20.0]),
            initial_soc=st.sampled_from([0.0, 0.5, 1.0]),
        )
    )
    site = InfrastructureSpec.wallboxes(
        points_per_station * stations,
        point_kw,
        points_per_station=points_per_station,
        station_power_kw=station_kw,
        transformer_power_kw=transformer_kw,
        storage=storage,
    )
    operation = Operation(
        scheduling=draw(st.sampled_from(["fcfs", "edf", "discrimination_free"])),
        queue_length=draw(st.integers(0, 3)),
        disconnect=draw(st.sampled_from(["at_departure", "when_full"])),
    )
    preload = draw(st.sampled_from([None, 0.1, 0.5]))
    return make_scenario(
        site,
        operation=operation,
        events_per_week=draw(st.integers(20, 400)),
        transformer_preload=None
        if preload is None
        else TimeSeries.constant(preload * transformer_kw),
    )


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenario=scenarios(), seed=st.integers(0, 10_000))
def test_limit_aware_policies_never_exceed_hardware_limits(scenario, seed):
    result = simulate(scenario, "2024-01-01", "2024-01-03 23:45", "15min", seed=seed)
    power = result.point_power_kw
    (transformer,) = scenario.infrastructure.transformers

    assert np.all(power >= -TOLERANCE)
    point_limits = np.array([p.max_power_kw for p in scenario.infrastructure.charging_points])
    assert np.all(power <= point_limits + TOLERANCE)

    column = 0
    for station in transformer.charging_stations:
        width = len(station.charging_points)
        assert np.all(
            power[:, column : column + width].sum(axis=1) <= station.max_power_kw + TOLERANCE
        )
        column += width

    assert np.all(result.grid_profile_kw <= transformer.max_power_kw + TOLERANCE)

    storage_soc_never_negative = result.storage_profile_kw.cumsum() * 0.25
    if scenario.infrastructure.transformers[0].storage is not None:
        spec = transformer.storage
        energy = spec.initial_soc * spec.capacity_kwh + storage_soc_never_negative
        assert np.all(energy >= spec.min_soc * spec.capacity_kwh - TOLERANCE)
        assert np.all(energy <= spec.capacity_kwh + TOLERANCE)


@settings(max_examples=25, deadline=None)
@given(scenario=scenarios(), seed=st.integers(0, 10_000))
def test_energy_delivered_never_exceeds_energy_demanded(scenario, seed):
    result = simulate(scenario, "2024-01-01", "2024-01-02 23:45", "15min", seed=seed)
    demanded = sum(
        (e.target_level - e.initial_level) * e.profile.battery.capacity_kwh
        for e in result.realisation.events
    )
    assert result.total_energy_kwh <= demanded + TOLERANCE
    assert result.rejections <= result.num_events


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_uncontrolled_is_an_upper_bound_on_energy(seed):
    base = make_scenario(
        InfrastructureSpec.wallboxes(4, 11, transformer_power_kw=15), events_per_week=300
    )
    realisation = base.realise(TimeGrid.create("2024-01-01", "2024-01-02 23:45", "15min"), seed)
    energy = {}
    for policy in ("uncontrolled", "fcfs", "edf", "discrimination_free"):
        variant = realisation.with_scenario(base.replace(operation=Operation(scheduling=policy)))
        energy[policy] = simulate(variant).total_energy_kwh
    assert energy["fcfs"] <= energy["uncontrolled"] + TOLERANCE
    assert energy["edf"] <= energy["uncontrolled"] + TOLERANCE
    assert energy["discrimination_free"] <= energy["uncontrolled"] + TOLERANCE
