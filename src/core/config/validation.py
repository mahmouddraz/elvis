"""Helpers that turn invalid inputs into readable :class:`ConfigurationError` messages."""

from __future__ import annotations

import math
import numbers
from collections.abc import Collection, Mapping
from typing import Any

from core.errors import ConfigurationError


class ProblemCollector:
    """Accumulates validation problems for one object and raises them together."""

    def __init__(self, context: str) -> None:
        self.context = context
        self.problems: list[str] = []

    def add(self, problem: str) -> None:
        self.problems.append(f"{self.context}: {problem}")

    def positive(self, name: str, value: float) -> None:
        if not _is_number(value) or value <= 0:
            self.add(f"{name} must be > 0, got {value!r}")

    def non_negative(self, name: str, value: float) -> None:
        if not _is_number(value) or value < 0:
            self.add(f"{name} must be >= 0, got {value!r}")

    def fraction(self, name: str, value: float) -> None:
        if not _is_number(value) or not 0 <= value <= 1:
            self.add(f"{name} must be in [0, 1], got {value!r}")

    def raise_if_any(self) -> None:
        if self.problems:
            raise ConfigurationError(self.problems)

    @staticmethod
    def single(problem: str) -> ConfigurationError:
        return ConfigurationError(problem)

    @staticmethod
    def reject_unknown_keys(
        context: str,
        raw: Mapping[str, Any],
        allowed: Collection[str],
        *,
        required: Collection[str] = (),
    ) -> None:
        """Fail on misspelled keys, which would otherwise be silently ignored."""
        if not isinstance(raw, Mapping):
            raise ConfigurationError(f"{context}: expected a mapping, got {raw!r}")
        problems = [f"{context}: unknown key {key!r}" for key in raw if key not in allowed]
        problems += [f"{context}: missing key {key!r}" for key in required if key not in raw]
        if problems:
            raise ConfigurationError(problems)


def _is_number(value: Any) -> bool:
    """Any finite real number, including numpy scalars; booleans are not numbers here."""
    return not isinstance(value, bool) and isinstance(value, numbers.Real) and math.isfinite(value)
