"""Small numeric helpers shared across the package."""

from __future__ import annotations

import math

# Values within this distance of a rounding boundary are treated as lying on it, so that
# float noise such as 0.37 -> 0.36999999999999994 does not lose a whole watt.
_NOISE = 1e-9


def floor_to(value: float, decimals: int = 3) -> float:
    """Round ``value`` down to ``decimals`` decimal places (watt precision for kW).

    Power assignments are floored so the sum of assigned powers never exceeds a limit.
    """
    factor = 10**decimals
    return math.floor(value * factor + _NOISE) / factor


def ceil_to(value: float, decimals: int = 3) -> float:
    """Round ``value`` up to ``decimals`` decimal places."""
    factor = 10**decimals
    return math.ceil(value * factor - _NOISE) / factor
