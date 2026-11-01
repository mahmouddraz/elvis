import datetime as dt

import numpy as np
import pytest

from core import ConfigurationError, TimeGrid, TimeSeries
from core.time import align_pandas_series

GRID = TimeGrid.create("2024-01-01", "2024-01-01 03:00", "01:00:00")


def test_constant():
    assert TimeSeries.from_config(5, name="x").align(GRID).tolist() == [5] * 4


def test_list_at_simulation_resolution():
    series = TimeSeries.from_config([1, 2, 3, 4, 5], name="x")
    assert series.align(GRID).tolist() == [1, 2, 3, 4]


def test_too_short_without_repeat():
    with pytest.raises(ConfigurationError, match="does not cover"):
        TimeSeries((1.0, 2.0)).align(GRID)


def test_repeat_wraps_periodically():
    series = TimeSeries((1.0, 2.0), repeat=True)
    assert series.align(GRID).tolist() == [1, 2, 1, 2]


def test_resampling_interpolates_linearly():
    quarter_hour_grid = TimeGrid.create("2024-01-01", "2024-01-01 01:00", "15min")
    series = TimeSeries((0.0, 4.0), resolution=dt.timedelta(hours=1))
    np.testing.assert_allclose(series.align(quarter_hour_grid), [0, 1, 2, 3, 4])


def test_repeat_with_resolution_interpolates_across_the_seam():
    grid = TimeGrid.create("2024-01-01", "2024-01-01 02:30", "30min")
    series = TimeSeries((0.0, 2.0), resolution=dt.timedelta(hours=1), repeat=True)
    np.testing.assert_allclose(series.align(grid), [0, 1, 2, 1, 0, 1])


def test_mapping_config_roundtrip():
    raw = {"values": [1.0, 2.0], "resolution": "01:00:00", "repeat": True}
    assert TimeSeries.from_config(raw, name="x").to_config() == raw


def test_rejects_bad_input():
    with pytest.raises(ConfigurationError):
        TimeSeries.from_config({"values": [1], "unknown": 1}, name="x")
    with pytest.raises(ConfigurationError):
        TimeSeries.from_config(["a"], name="x")
    with pytest.raises(ConfigurationError):
        TimeSeries((float("nan"),))


def test_pandas_series_alignment():
    pd = pytest.importorskip("pandas")
    index = pd.date_range("2024-01-01", periods=3, freq="2h")
    series = pd.Series([0.0, 2.0, 4.0], index=index)
    np.testing.assert_allclose(align_pandas_series(series, GRID), [0, 1, 2, 3])
    with pytest.raises(ConfigurationError):
        align_pandas_series(series.iloc[:1], GRID)
