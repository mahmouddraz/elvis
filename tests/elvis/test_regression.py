"""Seeded KPIs of every preset, pinned to detect unintended changes in simulation results.

If a change is *meant* to alter results (a model improvement or a bug fix), regenerate the
snapshot and explain the difference in the commit message::

    ELVIS_UPDATE_SNAPSHOT=1 uv run pytest tests/test_regression.py
"""

import json
import os
from pathlib import Path

import pytest

from elvis import list_presets, load_preset, simulate

SNAPSHOT = Path(__file__).parent / "data" / "regression.json"
PERIOD = ("2024-03-04", "2024-03-17 23:45", "00:15:00")
SEED = 20240304


def kpis(name):
    result = simulate(load_preset(name), *PERIOD, seed=SEED)
    summary = result.summary()
    summary["sessions"] = len(result.sessions)
    summary["storage_energy_kwh"] = float(abs(result.storage_profile_kw).sum() * 0.25)
    return summary


def test_snapshot():
    current = {name: kpis(name) for name in list_presets()}
    if os.environ.get("ELVIS_UPDATE_SNAPSHOT"):
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        pytest.skip("snapshot updated")
    expected = json.loads(SNAPSHOT.read_text())
    assert sorted(current) == sorted(expected), "presets were added or removed"
    for name, values in expected.items():
        for key, value in values.items():
            assert current[name][key] == pytest.approx(value, rel=1e-9, nan_ok=True), (name, key)
