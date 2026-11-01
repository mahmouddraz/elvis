"""Random-variate building blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.config.validation import ProblemCollector
from core.errors import ConfigurationError

Rng = np.random.Generator
"""All randomness flows through an explicit generator, so runs are reproducible from a seed."""


def make_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """A generator from a seed; generators (or compatible objects) pass through."""
    if seed is None or isinstance(seed, (int, np.integer)):
        return np.random.default_rng(seed)
    return seed


@dataclass(frozen=True, slots=True)
class TruncatedNormal:
    """A normal distribution truncated by clipping to ``[low, high]``.

    Clipping (rather than rejection) puts the excess probability mass on the bounds, e.g.
    vehicles that arrive with an exactly empty battery.
    """

    mean: float
    std: float
    low: float = -np.inf
    high: float = np.inf

    def __post_init__(self) -> None:
        if self.std < 0:
            raise ConfigurationError(f"std must be >= 0, got {self.std}")
        if self.low > self.high:
            raise ConfigurationError("low exceeds high")

    def sample(self, rng: Rng, size: int) -> np.ndarray:
        return np.clip(rng.normal(self.mean, self.std, size), self.low, self.high)

    @classmethod
    def from_config(cls, raw: dict[str, Any], context: str, **bounds: float) -> TruncatedNormal:
        """Parse ``{mean, std}``; the bounds are fixed by the caller."""
        ProblemCollector.reject_unknown_keys(
            context, raw, ("mean", "std"), required=("mean", "std")
        )
        return cls(float(raw["mean"]), float(raw["std"]), **bounds)

    def to_config(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}


class CategoricalSampler:
    """Draws category indices with given weights in O(1) per draw (Walker's alias method).

    Reference: A. J. Walker, "An efficient method for generating discrete random variables
    with general distributions", ACM TOMS 3(3), 1977.
    """

    def __init__(self, weights: Sequence[float]) -> None:
        scaled = np.asarray(weights, dtype=float)
        if scaled.ndim != 1 or len(scaled) == 0 or scaled.sum() <= 0 or np.any(scaled < 0):
            raise ConfigurationError("weights must be a non-empty vector of non-negative numbers")
        n = len(scaled)
        scaled = scaled * n / scaled.sum()
        alias = -np.ones(n, dtype=int)
        small = np.flatnonzero(scaled < 1).tolist()
        large = np.flatnonzero(scaled > 1).tolist()
        while small and large:
            j = small.pop()
            k = large[-1]
            alias[j] = k
            scaled[k] -= 1 - scaled[j]
            if scaled[k] < 1:
                small.append(k)
                large.pop()
        self._n = n
        self._threshold = scaled
        self._alias = alias

    def sample(self, rng: Rng, size: int) -> np.ndarray:
        """Draw ``size`` indices."""
        u = rng.random(size)
        j = rng.integers(self._n, size=size)
        return np.where(u <= self._threshold[j], j, self._alias[j])
