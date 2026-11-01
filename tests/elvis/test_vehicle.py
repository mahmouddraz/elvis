import pytest

from elvis import ConfigurationError, EVBattery, VehicleType


def test_power_is_constant_until_degradation_starts():
    battery = EVBattery(60, 100, start_power_degradation=0.8, max_degradation_level=0.2)
    assert battery.max_power_at(0.5) == 100
    assert battery.max_power_at(0.8) == 100
    assert battery.max_power_at(0.9) == pytest.approx(60)
    assert battery.max_power_at(1.0) == pytest.approx(20)


def test_no_degradation_by_default():
    assert EVBattery(60, 50).max_power_at(1.0) == 50


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity_kwh": 0, "max_charge_power_kw": 1},
        {"capacity_kwh": 1, "max_charge_power_kw": 1, "efficiency": 1.5},
        {"capacity_kwh": 1, "max_charge_power_kw": 1, "min_charge_power_kw": 2},
    ],
)
def test_battery_validation(kwargs):
    with pytest.raises(ConfigurationError):
        EVBattery(**kwargs)


def test_vehicle_config_roundtrip():
    vehicle = VehicleType("VW", "ID.3", EVBattery(58, 120), 0.3)
    assert VehicleType.from_config(vehicle.to_config()) == vehicle
    assert vehicle.name == "VW ID.3"


def test_config_rejects_misspelled_keys():
    raw = VehicleType("VW", "ID.3", EVBattery(58, 120), 0.3).to_config()
    raw["battery"]["max_power_degredation"] = 0.5
    with pytest.raises(ConfigurationError, match="max_power_degredation"):
        VehicleType.from_config(raw)


def test_config_reports_missing_keys():
    with pytest.raises(ConfigurationError, match="missing key 'battery'"):
        VehicleType.from_config({"brand": "VW", "model": "ID.3", "probability": 1})
