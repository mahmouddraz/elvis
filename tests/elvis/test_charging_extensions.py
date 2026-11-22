"""AC on-board charger limit, charging losses, and empirical demand from scenario files."""

import datetime as dt

import numpy as np
import pytest
import yaml

from elvis import (
    ChargingPointSpec,
    ChargingStationSpec,
    ConfigurationError,
    EmpiricalChargingDemand,
    EVBattery,
    InfrastructureSpec,
    MeasuredSession,
    Scenario,
    TimeGrid,
    TransformerSpec,
    VehicleType,
    simulate,
)
from elvis.demand import demand_from_config

from .conftest import MONDAY, make_event, make_scenario, realisation_of


def site(*points: ChargingPointSpec) -> InfrastructureSpec:
    total = sum(point.max_power_kw for point in points)
    station = ChargingStationSpec(total, tuple(points))
    return InfrastructureSpec((TransformerSpec(total, (station,)),))


def car(max_ac_power_kw=11.0, **battery) -> VehicleType:
    return VehicleType(
        "Test", "Car", EVBattery(50, 100, max_ac_power_kw=max_ac_power_kw, **battery), 1
    )


# -- AC on-board charger ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("point", "expected_kw"),
    [
        (ChargingPointSpec(22, type="ac"), 11),
        (ChargingPointSpec(150, type="dc"), 100),
        (ChargingPointSpec(150), 100),
        (ChargingPointSpec(7.4, type="ac"), 7.4),
    ],
)
def test_ac_limit_depends_on_point_type(point, expected_kw):
    scenario = make_scenario(site(point)).replace(vehicles=(car(),))
    events = [make_event(1, 0, 4, soc=0.0, vehicle=car())]
    result = simulate(realisation_of(scenario, events, hours=4, resolution="15min"))
    assert result.peak_load_kw == pytest.approx(expected_kw)


def test_without_ac_limit_ac_point_behaves_as_before():
    vehicle = car(max_ac_power_kw=None)
    scenario = make_scenario(site(ChargingPointSpec(22, type="ac"))).replace(vehicles=(vehicle,))
    result = simulate(realisation_of(scenario, [make_event(1, 0, 4, 0.0, vehicle)], hours=4))
    assert result.peak_load_kw == 22


def test_soc_taper_still_applies_at_ac_points():
    battery = EVBattery(
        50, 100, start_power_degradation=0.5, max_degradation_level=0.0, max_ac_power_kw=11
    )
    assert battery.max_ac_power_at(0.2) == 11
    assert battery.max_ac_power_at(0.95) == pytest.approx(10)
    assert battery.max_ac_power_at(1.0) == 0


def test_ac_fields_validate_and_roundtrip():
    with pytest.raises(ConfigurationError, match="type"):
        ChargingPointSpec(22, type="ccs")
    with pytest.raises(ConfigurationError, match="max_ac_power_kw"):
        EVBattery(50, 100, max_ac_power_kw=0)
    vehicle = car()
    assert VehicleType.from_config(vehicle.to_config()) == vehicle
    assert "max_ac_power_kw" not in car(max_ac_power_kw=None).to_config()["battery"]
    infrastructure = site(ChargingPointSpec(22, type="ac"), ChargingPointSpec(50, type="dc"))
    assert InfrastructureSpec.from_config(infrastructure.to_config()) == infrastructure
    kinds = [server.kind for server in infrastructure.to_network().servers]
    assert kinds == ["ac", "dc"]


# -- charging losses -------------------------------------------------------------------------


def run_single_session(efficiency: float):
    vehicle = VehicleType("Test", "Car", EVBattery(50, 11, efficiency=efficiency), 1)
    scenario = make_scenario(site(ChargingPointSpec(22))).replace(vehicles=(vehicle,))
    events = [make_event(1, 0, 10, soc=0.5, vehicle=vehicle)]
    return simulate(realisation_of(scenario, events, hours=10, resolution="15min"))


@pytest.mark.parametrize("efficiency", [0.9, 0.85, 0.5])
def test_grid_energy_is_battery_energy_over_efficiency(efficiency):
    lossless = run_single_session(1.0)
    lossy = run_single_session(efficiency)
    assert lossless.grid_energy_kwh == lossless.battery_energy_kwh == pytest.approx(25)
    assert lossy.battery_energy_kwh == pytest.approx(25, rel=1e-12)
    assert lossy.grid_energy_kwh == pytest.approx(25 / efficiency, rel=1e-12)
    assert lossy.charging_losses_kwh == pytest.approx(25 / efficiency - 25, rel=1e-9)
    assert lossy.peak_load_kw == pytest.approx(min(11 / efficiency, 22))
    summary = lossy.summary()
    assert summary["energy_kwh"] == summary["grid_energy_kwh"] == lossy.grid_energy_kwh
    assert summary["battery_energy_kwh"] == lossy.battery_energy_kwh


def test_point_limit_applies_on_grid_side():
    vehicle = VehicleType("Test", "Car", EVBattery(50, 11, efficiency=0.5), 1)
    scenario = make_scenario(site(ChargingPointSpec(11))).replace(vehicles=(vehicle,))
    result = simulate(realisation_of(scenario, [make_event(1, 0, 2, 0.0, vehicle)], hours=2))
    assert result.peak_load_kw == 11
    assert result.battery_energy_kwh == pytest.approx(11 * 2 * 0.5)


def test_zero_efficiency_is_rejected():
    with pytest.raises(ConfigurationError, match="efficiency"):
        EVBattery(50, 11, efficiency=0)


# -- empirical demand ------------------------------------------------------------------------

VEHICLES = (
    VehicleType("A", "Small", EVBattery(40, 11), 1),
    VehicleType("B", "Large", EVBattery(80, 11), 1),
)
SESSIONS = (
    MeasuredSession(8.0, 9.0, 12.0, "A Small"),
    MeasuredSession(18.5, 13.0, 30.0, "B Large"),
    MeasuredSession(40.25, 2.0, 5.0, "A Small"),
)


def empirical_scenario(sessions=SESSIONS, per_week=200.0) -> Scenario:
    return make_scenario().replace(
        vehicles=VEHICLES, demand=EmpiricalChargingDemand(per_week, tuple(sessions))
    )


def test_empirical_config_roundtrip(tmp_path):
    scenario = empirical_scenario()
    path = tmp_path / "site.yaml"
    scenario.to_yaml(path)
    assert yaml.safe_load(path.read_text())["demand"]["model"] == "empirical"
    assert Scenario.from_yaml(path) == scenario


def test_empirical_sessions_from_csv_relative_to_scenario_file(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sessions.csv").write_text(
        "arrival_hour_of_week,parking_hours,energy_kwh,vehicle\n"
        "8.0,9.0,12.0,A Small\n18.5,13.0,30.0,B Large\n40.25,2.0,5.0,A Small\n"
    )
    raw = empirical_scenario().to_config()
    raw["demand"] = {
        "model": "empirical",
        "events_per_week": 200,
        "sessions_csv": "data/sessions.csv",
    }
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(raw))
    loaded = Scenario.from_yaml(path)
    assert loaded.demand.sessions == SESSIONS
    assert loaded == empirical_scenario()
    assert Scenario.from_config(loaded.to_config()) == loaded


def test_empirical_preserves_the_joint_session_structure(week_grid):
    scenario = empirical_scenario()
    capacities = {vehicle.name: vehicle.capacity for vehicle in VEHICLES}
    allowed = {
        (
            s.arrival_hour_of_week,
            s.parking_hours,
            round(1 - s.energy_kwh / capacities[s.vehicle], 9),
            s.vehicle,
        )
        for s in SESSIONS
    }
    events = scenario.realise(week_grid, seed=3).events
    assert len(events) > 50
    seen = set()
    for event in events:
        hours = (event.arrival - MONDAY) / dt.timedelta(hours=1) % 168
        key = (hours, event.dwell_hours, round(event.initial_level, 9), event.profile.name)
        assert key in allowed
        seen.add(key)
    assert seen == allowed
    assert all(event.target_level == 1 for event in events)


def test_empirical_sessions_without_vehicle_draw_one(week_grid):
    sessions = (MeasuredSession(8.0, 9.0, 10.0),)
    events = empirical_scenario(sessions).realise(week_grid, seed=1).events
    names = {event.profile.name for event in events}
    assert names == {"A Small", "B Large"}
    for event in events:
        assert event.demand == pytest.approx(10)


def test_empirical_validation():
    with pytest.raises(ConfigurationError, match="unknown vehicle"):
        empirical_scenario((MeasuredSession(8.0, 9.0, 10.0, "C Missing"),))
    with pytest.raises(ConfigurationError, match="either sessions or sessions_csv"):
        demand_from_config({"model": "empirical", "events_per_week": 1})
    with pytest.raises(ConfigurationError, match="energy_kwh"):
        demand_from_config(
            {
                "model": "empirical",
                "events_per_week": 1,
                "sessions": [{"arrival_hour_of_week": 1, "parking_hours": 2}],
            }
        )
    with pytest.raises(ConfigurationError, match="hour of week"):
        empirical_scenario((MeasuredSession(170.0, 1.0, 1.0, "A Small"),)).system  # noqa: B018


def test_empirical_simulates(week_grid):
    result = simulate(empirical_scenario(), "2024-01-01", "2024-01-07 23:45", seed=2)
    assert result.num_events > 0 and result.grid_energy_kwh > 0


def test_ac_grid_draw_equals_on_board_charger_rating():
    vehicle = VehicleType("Test", "Car", EVBattery(50, 100, efficiency=0.9, max_ac_power_kw=11), 1)
    scenario = make_scenario(site(ChargingPointSpec(22, type="ac"))).replace(vehicles=(vehicle,))
    events = [make_event(1, 0, 2, soc=0.0, vehicle=vehicle)]
    result = simulate(realisation_of(scenario, events, hours=2, resolution="15min"))
    assert result.peak_load_kw == pytest.approx(11, rel=1e-12)
    assert result.battery_energy_kwh == pytest.approx(11 * 0.9 * 2, rel=1e-12)


def test_ac_taper_binds_on_battery_side():
    battery = EVBattery(50, 100, efficiency=0.9, start_power_degradation=0.5, max_ac_power_kw=11)
    assert battery.max_ac_power_at(0.2) == pytest.approx(9.9)
    assert battery.max_ac_power_at(0.97) == pytest.approx(6)


@pytest.mark.parametrize(("side", "battery_kwh"), [("grid", 9.0), ("battery", 10.0)])
def test_empirical_energy_side(tmp_path, side, battery_kwh):
    vehicle = VehicleType("A", "Small", EVBattery(40, 11, efficiency=0.9), 1)
    (tmp_path / "s.csv").write_text(
        "arrival_hour_of_week,parking_hours,energy_kwh,vehicle\n8.0,12.0,10.0,A Small\n"
    )
    raw = {"model": "empirical", "events_per_week": 3, "sessions_csv": "s.csv"}
    if side == "battery":
        raw["energy_side"] = side
    demand = demand_from_config(raw, tmp_path)
    scenario = make_scenario().replace(vehicles=(vehicle,), demand=demand)
    result = simulate(scenario, "2024-01-01", "2024-01-14 23:45", seed=4)
    events = result.realisation.events
    assert events
    for event in events:
        assert event.demand == pytest.approx(battery_kwh)
    outcomes = result.core.request_outcomes()
    assert outcomes["delivered"] == pytest.approx(outcomes["demand"])
    assert result.grid_energy_kwh == pytest.approx(len(events) * battery_kwh / 0.9)
    if side == "grid":
        assert result.grid_energy_kwh == pytest.approx(10 * len(events))
    assert Scenario.from_config(scenario.to_config()) == scenario


# -- conditional vehicle draw ----------------------------------------------------------------

FLEET = (
    VehicleType("S", "Small", EVBattery(20, 50, max_ac_power_kw=11), 1),
    VehicleType("M", "Mid", EVBattery(60, 50, max_ac_power_kw=11), 1),
    VehicleType("L", "Large", EVBattery(80, 100, max_ac_power_kw=22), 2),
)


def conditional_scenario(sessions, draw="conditional", point=None, per_week=300.0, eta=1.0):
    fleet = tuple(
        vehicle.__class__(
            vehicle.brand,
            vehicle.model,
            EVBattery(
                vehicle.capacity,
                vehicle.battery.max_charge_power_kw,
                efficiency=eta,
                max_ac_power_kw=vehicle.battery.max_ac_power_kw,
            ),
            vehicle.probability,
        )
        for vehicle in FLEET
    )
    infrastructure = InfrastructureSpec.wallboxes(40, 11) if point is None else site(point)
    demand = EmpiricalChargingDemand(per_week, tuple(sessions), vehicle_draw=draw)
    return make_scenario(infrastructure).replace(vehicles=fleet, demand=demand)


TWO_WEEKS = TimeGrid.create(MONDAY, MONDAY + dt.timedelta(weeks=2), "15min")


def drawn_names(scenario, seed=5):
    realisation = scenario.realise(TWO_WEEKS, seed=seed)
    return [event.profile.name for event in realisation.events], realisation.draw_stats


def test_infeasible_vehicle_is_never_drawn_and_fleet_ratio_is_kept():
    names, stats = drawn_names(conditional_scenario([MeasuredSession(8.0, 4.0, 30.0)]))
    assert "S Small" not in names
    assert stats["drawn_conditionally"] == stats["sessions_without_profile"] == len(names)
    assert stats["infeasible"] == 0
    share = names.count("L Large") / len(names)
    standard_error = np.sqrt(2 / 3 * 1 / 3 / len(names))
    assert share == pytest.approx(2 / 3, abs=4 * standard_error)


def test_ac_power_limit_decides_feasibility():
    point = ChargingPointSpec(22, type="ac")
    names, _ = drawn_names(conditional_scenario([MeasuredSession(8.0, 1.5, 30.0)], point=point))
    assert set(names) == {"L Large"}


def test_infeasible_sessions_fall_back_to_the_full_mix_and_are_counted():
    names, stats = drawn_names(conditional_scenario([MeasuredSession(8.0, 1.0, 30.0)]))
    assert stats["infeasible"] == len(names) > 0
    assert stats["drawn_conditionally"] == 0
    assert "S Small" in names


def test_independent_draw_is_unchanged():
    sessions = [MeasuredSession(8.0, 4.0, 30.0)]
    independent = conditional_scenario(sessions, draw="independent")
    names, stats = drawn_names(independent)
    assert "S Small" in names and stats["drawn_conditionally"] == 0
    base = make_scenario().replace(
        vehicles=independent.vehicles, demand=EmpiricalChargingDemand(300.0, tuple(sessions))
    )
    assert drawn_names(base)[0] == names


def test_conditional_session_delivers_its_metered_grid_energy():
    scenario = conditional_scenario([MeasuredSession(8.0, 4.0, 30.0)], per_week=3, eta=0.9)
    result = simulate(scenario, "2024-01-01", "2024-01-14 23:45", seed=7)
    assert result.num_events > 0 and result.rejections == 0
    assert result.grid_energy_kwh == pytest.approx(30 * result.num_events, rel=1e-12)
    assert result.draw_stats["drawn_conditionally"] == result.num_events


def test_vehicle_draw_config():
    scenario = conditional_scenario([MeasuredSession(8.0, 4.0, 30.0)])
    assert scenario.to_config()["demand"]["vehicle_draw"] == "conditional"
    assert Scenario.from_config(scenario.to_config()) == scenario
    assert "vehicle_draw" not in empirical_scenario().to_config()["demand"]
    with pytest.raises(ConfigurationError, match="vehicle_draw"):
        EmpiricalChargingDemand(1.0, SESSIONS, vehicle_draw="smart")


def test_vehicle_without_dc_inlet_gets_no_power_at_dc_points():
    battery = EVBattery(17, 22, max_ac_power_kw=22, dc_capable=False)
    vehicle = VehicleType("Smart", "EQ", battery, 1)
    for point, served in [
        (ChargingPointSpec(50, type="dc"), False),
        (ChargingPointSpec(22, type="ac"), True),
        (ChargingPointSpec(50), True),
    ]:
        scenario = make_scenario(site(point)).replace(vehicles=(vehicle,))
        result = simulate(realisation_of(scenario, [make_event(1, 0, 2, 0.5, vehicle)], hours=2))
        assert (result.grid_energy_kwh > 0) == served
        assert result.unservable_events == (0 if served else 1)
    assert VehicleType.from_config(vehicle.to_config()) == vehicle
    assert "dc_capable" not in car().to_config()["battery"]


def test_vehicle_without_dc_inlet_is_rejected_when_full_and_does_not_block_the_point():
    from elvis import Operation

    smart = VehicleType("Smart", "EQ", EVBattery(17, 22, max_ac_power_kw=22, dc_capable=False), 1)
    scenario = make_scenario(
        site(ChargingPointSpec(50, type="dc")),
        operation=Operation(queue_length=10, disconnect="when_full"),
    ).replace(vehicles=(smart, car()))
    events = [make_event(1, 0, 24, 0.5, smart), make_event(2, 1, 24, 0.5, car())]
    result = simulate(realisation_of(scenario, events, hours=6, resolution="15min"))
    assert result.unservable_events == 1
    assert result.rejections == 1
    assert 1 not in result.core.delivered
    assert result.core.delivered[2] > 0


def test_conditional_draw_excludes_vehicles_without_dc_inlet_at_dc_sites():
    fleet = (
        VehicleType("Smart", "EQ", EVBattery(17, 22, max_ac_power_kw=22, dc_capable=False), 3),
        VehicleType("L", "Large", EVBattery(80, 100), 1),
    )
    demand = EmpiricalChargingDemand(
        300.0, (MeasuredSession(8.0, 2.0, 10.0),), vehicle_draw="conditional"
    )
    dc = make_scenario(site(ChargingPointSpec(50, type="dc"))).replace(
        vehicles=fleet, demand=demand
    )
    names, stats = drawn_names(dc)
    assert set(names) == {"L Large"} and stats["infeasible"] == 0
    ac = dc.replace(infrastructure=site(ChargingPointSpec(22, type="ac")))
    assert "Smart EQ" in drawn_names(ac)[0]


# -- energy from SOC -------------------------------------------------------------------------


def soc_raw(**extra):
    return {
        "model": "empirical",
        "events_per_week": 400,
        "energy_from": "soc",
        "soc_arrival": {"mean": 0.3, "std": 0.1},
        "sessions": [
            {"arrival_hour_of_week": 8.0, "parking_hours": 9.0, "energy_kwh": 999.0},
            {"arrival_hour_of_week": 18.5, "parking_hours": 13.0},
        ],
        **extra,
    }


def test_energy_from_soc_ignores_energy_and_keeps_arrival_dwell_pairs(week_grid):
    scenario = make_scenario().replace(vehicles=VEHICLES, demand=demand_from_config(soc_raw()))
    events = scenario.realise(week_grid, seed=8).events
    pairs = set()
    for event in events:
        hours = (event.arrival - MONDAY) / dt.timedelta(hours=1) % 168
        pairs.add((hours, event.dwell_hours))
        assert event.demand <= event.profile.capacity
    assert pairs == {(8.0, 9.0), (18.5, 13.0)}
    levels = np.array([event.initial_level for event in events])
    standard_error = 0.1 / np.sqrt(len(levels))
    assert levels.mean() == pytest.approx(0.3, abs=4 * standard_error)
    assert levels.std() == pytest.approx(0.1, rel=0.15)
    assert {event.profile.name for event in events} == {"A Small", "B Large"}


def test_energy_from_soc_config():
    demand = demand_from_config(soc_raw(soc_target={"mean": 0.9, "std": 0.05}))
    assert demand_from_config(demand.to_config()) == demand
    assert "energy_from" not in EmpiricalChargingDemand(1.0, SESSIONS).to_config()
    with pytest.raises(ConfigurationError, match="vehicle_draw"):
        demand_from_config(soc_raw(vehicle_draw="conditional"))
    with pytest.raises(ConfigurationError, match="energy_kwh"):
        demand_from_config({**soc_raw(), "energy_from": "sessions", "soc_arrival": None})
