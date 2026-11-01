import pytest

from core.network import Network
from elvis import ConfigurationError, InfrastructureSpec
from elvis.infrastructure import (
    ChargingPointSpec,
    ChargingStationSpec,
    StorageSpec,
    TransformerSpec,
)


def test_wallboxes_defaults():
    spec = InfrastructureSpec.wallboxes(6, 11, points_per_station=2)
    (transformer,) = spec.transformers
    assert len(transformer.charging_stations) == 3
    assert transformer.charging_stations[0].max_power_kw == 22
    assert transformer.max_power_kw == 66
    assert spec.nominal_power_kw == 66


def test_wallboxes_rejects_uneven_split():
    with pytest.raises(ConfigurationError):
        InfrastructureSpec.wallboxes(5, 11, points_per_station=2)


def test_count_expands_and_compacts():
    raw = {
        "transformers": [
            {
                "max_power_kw": 100,
                "charging_stations": [
                    {
                        "max_power_kw": 22,
                        "count": 3,
                        "charging_points": [{"max_power_kw": 11, "count": 2}],
                    },
                    {"max_power_kw": 50, "charging_points": [{"max_power_kw": 50}]},
                ],
            }
        ]
    }
    spec = InfrastructureSpec.from_config(raw)
    assert len(spec.charging_points) == 7
    assert spec.to_config() == raw
    assert InfrastructureSpec.from_config(spec.to_config()) == spec


def test_storage_roundtrip():
    storage = StorageSpec(capacity_kwh=50, max_power_kw=10, initial_soc=0.5)
    spec = InfrastructureSpec.wallboxes(2, 11, storage=storage)
    assert InfrastructureSpec.from_config(spec.to_config()) == spec


@pytest.mark.parametrize(
    "build",
    [
        lambda: ChargingPointSpec(-1),
        lambda: ChargingPointSpec(10, min_power_kw=20),
        lambda: ChargingStationSpec(10, ()),
        lambda: StorageSpec(capacity_kwh=10, max_power_kw=5, initial_soc=0.1, min_soc=0.2),
        lambda: InfrastructureSpec(()),
    ],
)
def test_invalid_specs(build):
    with pytest.raises(ConfigurationError):
        build()


def test_unknown_key_is_reported():
    with pytest.raises(ConfigurationError, match="unknown key 'max_power'"):
        InfrastructureSpec.from_config(
            {"transformers": [{"max_power": 1, "max_power_kw": 1, "charging_stations": []}]}
        )


def test_maps_onto_a_core_network():
    spec = InfrastructureSpec(
        (
            TransformerSpec(
                100,
                (ChargingStationSpec(22, (ChargingPointSpec(11),) * 2),) * 2,
                storage=StorageSpec(capacity_kwh=50, max_power_kw=10),
            ),
            TransformerSpec(50, (ChargingStationSpec(50, (ChargingPointSpec(50),)),)),
        )
    )
    network = Network(spec.to_network())
    assert [s.id for s in network.servers] == ["cp-1", "cp-2", "cp-3", "cp-4", "cp-5"]
    assert [r.id for r in network.roots] == ["transformer-1", "transformer-2"]
    assert network.servers[0].parent.id == "t1-station-1"
    assert network.servers[4].root is network.roots[1]
    assert [b.id for b in network.buffers] == ["storage-1"]
    assert network.roots[0].max_rate == 100
    assert spec.to_network().nominal_rate == spec.nominal_power_kw == 94
