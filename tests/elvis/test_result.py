import numpy as np
import pytest

from elvis import ConfigurationError, InfrastructureSpec, TimeSeries, simulate

from .conftest import make_event, make_scenario, realisation_of


@pytest.fixture
def result():
    """Two 11 kW points; one vehicle charges 11 kW from 01:00 to 03:00 on a 30 min grid."""
    scenario = make_scenario(
        InfrastructureSpec.wallboxes(2, 11),
        transformer_preload=TimeSeries.constant(3),
        emission_factors=TimeSeries.constant(400),
    )
    events = [make_event(1, 1, 2, soc=0.0)]
    return simulate(realisation_of(scenario, events, hours=4, resolution="30min"))


def test_energy_and_peak(result):
    assert result.total_energy_kwh == pytest.approx(22)
    assert result.peak_load_kw == 11
    assert result.nominal_power_kw == 22
    assert result.simultaneity_factor() == 0.5
    assert result.simultaneity_factor(quantile=0) == 0


def test_profiles_and_dataframe(result):
    frame = result.to_dataframe()
    assert list(frame.columns) == ["cp-1", "cp-2", "charging_total", "grid_total"]
    assert len(frame) == 9
    np.testing.assert_allclose(frame["grid_total"] - frame["charging_total"], 3)


def test_charging_time(result):
    np.testing.assert_allclose(result.charging_hours(), [2.0])  # four 30 min steps
    assert result.mean_charging_hours() == 2.0


def test_costs_and_emissions(result):
    assert result.energy_cost(0.3) == pytest.approx(6.6)
    flat_curve = [0.3] * 24
    assert result.energy_cost(flat_curve) == pytest.approx(6.6)
    night_free = [0.0] * 2 + [1.0] * 22  # free 00:00-02:00, ramping to 1.0 by 02:00
    assert result.energy_cost(night_free) == pytest.approx(11 * 0.5 * (0 + 0.5 + 1 + 1))
    assert result.emissions_kg() == pytest.approx(22 * 0.4)


def test_emissions_require_factors():
    scenario = make_scenario()
    result = simulate(realisation_of(scenario, []))
    with pytest.raises(ConfigurationError):
        result.emissions_kg()
    assert np.isnan(result.mean_charging_hours())


def test_summary(result):
    summary = result.summary()
    assert summary["energy_kwh"] == pytest.approx(22)
    assert summary["events"] == 1 and summary["rejections"] == 0
    assert "test" in repr(result)
