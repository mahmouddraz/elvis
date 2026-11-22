"""Fixed versus Poisson-distributed arrival counts."""

import datetime as dt

import numpy as np
import pytest

from core import ConfigurationError, TimeGrid, TruncatedNormal
from core.arrivals import (
    GaussianMixtureArrivals,
    MixtureComponent,
    WeeklyProfileArrivals,
    arrival_process_from_config,
)

from .conftest import MONDAY

WEEK = TimeGrid.create(MONDAY, MONDAY + dt.timedelta(days=7) - dt.timedelta(hours=1), "1h")
SEEDS = range(4000)


def weekly(arrival_count="fixed", per_week=20.0):
    return WeeklyProfileArrivals(per_week, (1.0,) * 168, TruncatedNormal(3, 1, 0, 6), arrival_count)


def gmm(arrival_count="fixed", per_week=20):
    component = MixtureComponent(1.0, (80.0, 4.0), ((4.0, 0.0), (0.0, 1.0)))
    return GaussianMixtureArrivals(per_week, (component,), arrival_count=arrival_count)


def counts(process):
    return np.array([len(process.sample(WEEK, np.random.default_rng(s))[0]) for s in SEEDS])


def test_weekly_poisson_count_has_poisson_mean_and_variance():
    process = weekly("poisson")
    expected = 20.0 * 167 / 168
    sample = counts(process)
    standard_error = np.sqrt(expected / len(sample))
    assert sample.mean() == pytest.approx(expected, abs=4 * standard_error)
    assert sample.var(ddof=1) == pytest.approx(expected, rel=0.1)


def test_weekly_fixed_count_is_unchanged():
    assert set(counts(weekly())[:50]) == {20}


def test_gmm_poisson_count_has_poisson_mean_and_variance():
    sample = counts(gmm("poisson"))
    assert sample.mean() == pytest.approx(20, abs=4 * np.sqrt(20 / len(sample)))
    assert sample.var(ddof=1) == pytest.approx(20, rel=0.1)
    assert set(counts(gmm())[:50]) == {20}


def test_arrival_count_config():
    for process in (weekly("poisson"), gmm("poisson")):
        assert arrival_process_from_config(process.to_config()) == process
    assert "arrival_count" not in weekly().to_config()
    assert "arrival_count" not in gmm().to_config()
    with pytest.raises(ConfigurationError, match="arrival_count"):
        weekly("binomial")
