import json

import numpy as np
import pytest
import yaml

from elvis import (
    ConfigurationError,
    OpeningHours,
    Operation,
    Realisation,
    Scenario,
    TimeGrid,
    list_presets,
    load_preset,
    simulate,
)
from elvis.cli import main

from .conftest import make_scenario


def test_yaml_roundtrip(tmp_path):
    scenario = make_scenario(operation=Operation("fcfs", 2, "when_full", OpeningHours(6, 22)))
    path = tmp_path / "site.yaml"
    scenario.to_yaml(path)
    assert Scenario.from_yaml(path) == scenario


def test_all_problems_are_reported_at_once(tmp_path):
    raw = make_scenario().to_config()
    raw["operation"]["queue_lenght"] = 3
    raw["vehicles"][0]["battery"]["capacity_kwh"] = -1
    del raw["demand"]["soc_arrival"]
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigurationError) as info:
        Scenario.from_yaml(path)
    assert str(path) in str(info.value)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("operation", "queue_lenght", 1, "unknown key 'queue_lenght'"),
        ("operation", "scheduling", "UC", "policy must be one of"),
        ("operation", "disconnect", "never", "disconnect must be"),
        ("operation", "opening_hours", [20, 8], "opening_hours"),
        (None, "schema_version", 99, "schema_version"),
    ],
)
def test_invalid_values(section, key, value, message):
    raw = make_scenario().to_config()
    (raw[section] if section else raw)[key] = value
    with pytest.raises(ConfigurationError, match=message):
        Scenario.from_config(raw)


def test_duplicate_vehicle_names_are_rejected():
    scenario = make_scenario()
    with pytest.raises(ConfigurationError, match="unique"):
        scenario.replace(vehicles=scenario.vehicles * 2)


def test_replace_does_not_mutate():
    scenario = make_scenario()
    variant = scenario.replace(name="variant")
    assert scenario.name == "test" and variant.name == "variant"


@pytest.mark.parametrize("name", list_presets())
def test_every_preset_loads_roundtrips_and_simulates(name, tmp_path):
    scenario = load_preset(name)
    scenario.to_yaml(tmp_path / "copy.yaml")
    assert Scenario.from_yaml(tmp_path / "copy.yaml") == scenario
    result = simulate(scenario, "2024-03-04", "2024-03-10 23:45", seed=0)
    assert result.total_energy_kwh > 0
    assert 0 <= result.rejection_rate < 1


def test_unknown_preset():
    with pytest.raises(ConfigurationError, match="available"):
        load_preset("airport")


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_realisation_save_and_load(tmp_path, suffix):
    scenario = make_scenario(events_per_week=300)
    realisation = scenario.realise(TimeGrid.create("2024-01-01", "2024-01-07", "15min"), seed=5)
    path = tmp_path / f"realisation{suffix}"
    realisation.save(path)
    loaded = Realisation.load(path)
    assert loaded.events == realisation.events
    assert loaded.seed == 5
    np.testing.assert_array_equal(
        simulate(loaded).load_profile_kw, simulate(realisation).load_profile_kw
    )


def test_cli(tmp_path, capsys):
    assert main(["presets"]) == 0
    assert "office" in capsys.readouterr().out

    csv = tmp_path / "out.csv"
    code = main(
        [
            "run",
            "office",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-02",
            "--seed",
            "1",
            "--csv",
            str(csv),
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["scenario"] == "office"
    assert csv.read_text().startswith("time,cp-1")

    assert main(["run", "nowhere", "--start", "2024-01-01", "--end", "2024-01-02"]) == 2
    assert "unknown preset" in capsys.readouterr().err


def test_soc_target_roundtrip_and_effect():
    raw = make_scenario().to_config()
    raw["demand"]["soc_target"] = {"mean": 0.8, "std": 0.0}
    scenario = Scenario.from_config(raw)
    assert Scenario.from_config(scenario.to_config()) == scenario
    result = simulate(scenario, "2024-01-01", "2024-01-07 23:45", seed=0)
    assert {e.target_level for e in result.realisation.events} == {0.8}
