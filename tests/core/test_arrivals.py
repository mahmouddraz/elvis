import datetime as dt

import numpy as np
import pytest

from core import ConfigurationError, Demand, EmpiricalDemand, TimeGrid, TruncatedNormal
from core.arrivals import (
    GaussianMixtureArrivals,
    MixtureComponent,
    WeeklyProfileArrivals,
    arrival_process_from_config,
)
from core.arrivals.gaussian_mixture import _unshift_within_day
from core.random import CategoricalSampler
from core.time import hour_of_week

from .conftest import MONDAY, Tank

FOUR_WEEKS = TimeGrid.create(MONDAY, MONDAY + dt.timedelta(weeks=4), "15min")


def weekly(profile=(1.0,) * 168, per_week=500, dwell=None):
    return WeeklyProfileArrivals(per_week, tuple(profile), dwell or TruncatedNormal(3, 1, 0, 6))


def demand(arrivals, profiles=None, weights=(1.0,)):
    profiles = profiles or (Tank(),)
    return Demand(arrivals, TruncatedNormal(0.5, 0.2, 0, 1), tuple(profiles), tuple(weights))


class TestCategoricalSampler:
    def test_frequencies_match_weights(self):
        draws = CategoricalSampler([1, 2, 3, 4]).sample(np.random.default_rng(0), 200_000)
        np.testing.assert_allclose(
            np.bincount(draws) / len(draws), [0.1, 0.2, 0.3, 0.4], atol=0.005
        )

    def test_zero_weight_never_drawn(self):
        assert set(CategoricalSampler([0, 1]).sample(np.random.default_rng(0), 10_000)) == {1}

    def test_rejects_invalid_weights(self):
        with pytest.raises(ConfigurationError):
            CategoricalSampler([0, 0])


class TestWeeklyProfile:
    def test_expected_number_of_arrivals(self):
        arrivals, dwell = weekly().sample(FOUR_WEEKS, np.random.default_rng(0))
        assert len(arrivals) == len(dwell) == 2000

    def test_arrivals_follow_the_profile(self):
        mornings = [8.0 if d < 5 and 7 <= h < 10 else 0.0 for d in range(7) for h in range(24)]
        arrivals, _ = weekly(mornings).sample(FOUR_WEEKS, np.random.default_rng(1))
        hours = np.array([hour_of_week(a) for a in arrivals])
        # Linear interpolation spreads weight up to one slot before and after the window.
        assert np.all((hours % 24 >= 6) & (hours % 24 <= 10))
        assert np.all(hours < 5 * 24)

    def test_probabilities_interpolate_and_sum_to_one(self):
        profile = [0.0] * 168
        profile[1] = 1.0
        grid = TimeGrid.create(MONDAY, MONDAY + dt.timedelta(hours=2), "30min")
        np.testing.assert_allclose(
            weekly(profile).step_probabilities(grid), [0, 0.25, 0.5, 0.25, 0]
        )

    def test_dwell_is_truncated(self):
        _, dwell = weekly(dwell=TruncatedNormal(1, 5, 0, 2)).sample(
            FOUR_WEEKS, np.random.default_rng(0)
        )
        assert dwell.min() >= 0 and dwell.max() <= 2

    def test_config_roundtrip_and_validation(self):
        process = weekly()
        assert arrival_process_from_config(process.to_config()) == process
        with pytest.raises(ConfigurationError, match="process must be one of"):
            arrival_process_from_config({"process": "poisson"})
        with pytest.raises(ConfigurationError):
            weekly(profile=(1.0,) * 7 * 25)  # 175 slots do not split a week into whole minutes


class TestGaussianMixture:
    @staticmethod
    def mixture(**kwargs):
        components = (
            MixtureComponent(0.5, (9.0, 8.0), ((1.0, 0.0), (0.0, 1.0))),
            MixtureComponent(0.5, (33.0, 2.0), ((1.0, 0.0), (0.0, 0.5))),
        )
        return GaussianMixtureArrivals(100, components, **kwargs)

    def test_arrivals_within_period_and_on_grid(self):
        grid = TimeGrid.create(
            MONDAY + dt.timedelta(days=2), MONDAY + dt.timedelta(days=16), "15min"
        )
        arrivals, _ = self.mixture().sample(grid, np.random.default_rng(0))
        assert arrivals
        assert arrivals == sorted(arrivals)
        assert all(grid.start <= a <= grid.timestamps[-1] for a in arrivals)
        assert all(a.minute % 15 == 0 and a.second == 0 for a in arrivals)

    def test_short_dwell_is_resampled_long_dwell_clipped(self):
        _, dwell = self.mixture(min_dwell_hours=1.5, max_dwell_hours=9).sample(
            FOUR_WEEKS, np.random.default_rng(0)
        )
        assert dwell.min() >= 1.5 and dwell.max() <= 9

    def test_day_shift(self):
        hours = np.array([-1.0, 0.0, 18.0, 19.5, 23.9, 24.0, 167.0, 170.0])
        np.testing.assert_allclose(
            _unshift_within_day(hours, 5), [4.0, 5.0, 23.0, 0.5, 4.9, 29.0, 148.0, 170.0]
        )

    def test_config_roundtrip_and_validation(self):
        process = self.mixture()
        assert arrival_process_from_config(process.to_config()) == process
        bad = process.to_config() | {"components": process.to_config()["components"][:1]}
        with pytest.raises(ConfigurationError, match="sum to 1"):
            arrival_process_from_config(bad)


class TestDemand:
    def test_requests_are_numbered_and_sorted(self):
        requests = demand(weekly()).generate(FOUR_WEEKS, np.random.default_rng(0))
        assert [r.id for r in requests] == list(range(1, len(requests) + 1))
        assert [r.arrival for r in requests] == sorted(r.arrival for r in requests)
        assert all(0 <= r.initial_level <= 1 for r in requests)

    def test_same_seed_same_requests(self):
        model = demand(weekly())
        assert model.generate(FOUR_WEEKS, np.random.default_rng(7)) == model.generate(
            FOUR_WEEKS, np.random.default_rng(7)
        )

    def test_profile_mix(self):
        common, rare = Tank("common"), Tank("rare")
        requests = demand(weekly(per_week=5000), (common, rare), (0.9, 0.1)).generate(
            FOUR_WEEKS, np.random.default_rng(0)
        )
        assert sum(r.profile is rare for r in requests) / len(requests) == pytest.approx(
            0.1, abs=0.01
        )

    def test_validation(self):
        with pytest.raises(ConfigurationError, match="one weight per profile"):
            demand(weekly(), (Tank(),), (1.0, 2.0))
        with pytest.raises(ConfigurationError, match="unique"):
            demand(weekly(), (Tank(), Tank()), (1.0, 1.0))


class TestEmpiricalDemand:
    @staticmethod
    def model(rate=200):
        sessions = np.array([[18.0, 12.0, 10.0, 0], [8.5, 4.0, 5.0, 1], [100.0, 2.0, 40.0, 0]])
        return EmpiricalDemand(rate, sessions, (Tank("small", 20), Tank("large", 80)))

    def test_sessions_are_resampled_intact(self):
        requests = self.model().generate(FOUR_WEEKS, np.random.default_rng(0))
        observed = {(r.dwell_hours, round(r.demand, 6), r.profile.name) for r in requests}
        assert observed <= {(12.0, 10.0, "small"), (4.0, 5.0, "large"), (2.0, 20.0, "small")}
        assert {hour_of_week(r.arrival) for r in requests} <= {18.0, 8.5, 100.0}

    def test_volume_period_and_grid(self):
        requests = self.model().generate(FOUR_WEEKS, np.random.default_rng(0))
        weeks = FOUR_WEEKS.duration_hours / 168
        assert abs(len(requests) - 200 * weeks) < 4 * np.sqrt(200 * weeks)
        assert all(FOUR_WEEKS.start <= r.arrival <= FOUR_WEEKS.timestamps[-1] for r in requests)
        assert [r.arrival for r in requests] == sorted(r.arrival for r in requests)
        assert all(r.arrival.minute % 15 == 0 for r in requests)

    def test_quantity_above_capacity_is_capped(self):
        requests = self.model().generate(FOUR_WEEKS, np.random.default_rng(1))
        assert all(0 <= r.initial_level <= 1 for r in requests)

    def test_validation(self):
        with pytest.raises(ConfigurationError, match="profile index"):
            EmpiricalDemand(10, np.array([[1.0, 1.0, 1.0, 5]]), (Tank(),))
        with pytest.raises(ConfigurationError, match="hour of week"):
            EmpiricalDemand(10, np.array([[200.0, 1.0, 1.0, 0]]), (Tank(),))


def test_target_level_distribution():
    model = Demand(
        weekly(),
        TruncatedNormal(0.3, 0.1, 0, 1),
        (Tank(),),
        (1.0,),
        target_level=TruncatedNormal(0.8, 0.05, 0, 1),
    )
    requests = model.generate(FOUR_WEEKS, np.random.default_rng(0))
    targets = np.array([r.target_level for r in requests])
    assert targets.mean() == pytest.approx(0.8, abs=0.01)
    assert all(r.initial_level <= r.target_level for r in requests)


def test_empirical_demand_rejects_invalid_profiles():
    with pytest.raises(ConfigurationError, match="capacity"):
        EmpiricalDemand(10, np.array([[1.0, 1.0, 1.0, 0]]), (Tank(capacity=float("nan")),))
