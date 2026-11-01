import datetime as dt

import pytest

from core import ConfigurationError, TimeGrid
from core.time import format_duration, hour_of_week, parse_duration


def test_grid_counts_inclusive_end():
    grid = TimeGrid.create("2024-01-01", "2024-01-01 23:45", "00:15:00")
    assert grid.num_steps == 96
    assert grid.step_hours == 0.25
    assert grid.timestamps[-1] == dt.datetime(2024, 1, 1, 23, 45)
    assert grid.duration_hours == 23.75


def test_grid_end_off_grid_is_truncated():
    grid = TimeGrid.create("2024-01-01", "2024-01-01 01:10", "30min")
    assert grid.timestamps[-1] == dt.datetime(2024, 1, 1, 1, 0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("00:15:00", dt.timedelta(minutes=15)),
        ("15min", dt.timedelta(minutes=15)),
        ("36:00:00", dt.timedelta(hours=36)),
        ("1h", dt.timedelta(hours=1)),
    ],
)
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["abc", "00:00:00", "-1h"])
def test_parse_duration_rejects_invalid(text):
    with pytest.raises(ConfigurationError):
        parse_duration(text)


def test_format_duration_roundtrip():
    assert parse_duration(format_duration(dt.timedelta(hours=26, seconds=5))) == dt.timedelta(
        hours=26, seconds=5
    )


def test_invalid_grid():
    with pytest.raises(ConfigurationError):
        TimeGrid.create("2024-01-02", "2024-01-01", "1h")
    with pytest.raises(ConfigurationError):
        TimeGrid.create("not a date", "2024-01-01", "1h")


def test_hour_of_week():
    assert hour_of_week(dt.datetime(2024, 1, 1)) == 0  # a Monday
    assert hour_of_week(dt.datetime(2024, 1, 7, 23, 30)) == 167.5


def test_step_of():
    grid = TimeGrid.create("2024-01-01", "2024-01-02", "1h")
    assert grid.step_of(dt.datetime(2024, 1, 1, 2)) == 2
    assert grid.step_of(dt.datetime(2024, 1, 1, 2, 1)) == 3


@pytest.mark.parametrize(
    ("text", "hours"),
    [("1h 30min", 1.5), ("2 days", 48), ("90s", 0.025), ("1.5h", 1.5), ("1w", 168)],
)
def test_parse_duration_units(text, hours):
    assert parse_duration(text) == dt.timedelta(hours=hours)


@pytest.mark.parametrize("text", ["15", "1 month", "1h foo", "h"])
def test_parse_duration_rejects_unknown_units(text):
    with pytest.raises(ConfigurationError):
        parse_duration(text)
