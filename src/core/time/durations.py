"""Parsing and formatting of user-supplied datetimes and durations (no pandas required)."""

from __future__ import annotations

import datetime as dt
import re

from core.errors import ConfigurationError

_UNIT_SECONDS = {
    "s": 1, "sec": 1, "second": 1, "seconds": 1,
    "min": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
    "w": 604800, "week": 604800, "weeks": 604800,
}  # fmt: skip
_TERM = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([a-z]+)\s*")


def parse_datetime(value: str | dt.datetime, *, name: str = "datetime") -> dt.datetime:
    """Parse an ISO 8601 string (``"2024-01-01 06:30"``) or pass a datetime through."""
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value)
        except ValueError as error:
            raise ConfigurationError(f"{name}: {value!r} is not an ISO 8601 datetime") from error
    raise ConfigurationError(f"{name}: expected str or datetime, got {type(value).__name__}")


def parse_duration(value: str | dt.timedelta, *, name: str = "duration") -> dt.timedelta:
    """Parse a positive duration.

    Accepts a timedelta, ``"HH:MM:SS"`` (hours may exceed 23, e.g. ``"36:00:00"``) or unit
    terms such as ``"15min"``, ``"1h"`` or ``"1h 30min"``.
    """
    if isinstance(value, dt.timedelta):
        duration = value
    elif isinstance(value, str):
        duration = _parse_duration_string(value.strip().lower(), name)
    else:
        raise ConfigurationError(f"{name}: expected str or timedelta, got {type(value).__name__}")
    if duration <= dt.timedelta(0):
        raise ConfigurationError(f"{name}: must be positive, got {value!r}")
    return duration


def _parse_duration_string(text: str, name: str) -> dt.timedelta:
    parts = text.split(":")
    if len(parts) == 3:
        try:
            hours, minutes, seconds = (float(part) for part in parts)
        except ValueError:
            pass
        else:
            return dt.timedelta(hours=hours, minutes=minutes, seconds=seconds)
    seconds_total = 0.0
    position = 0
    for match in _TERM.finditer(text):
        number, unit = match.groups()
        if match.start() != position or unit not in _UNIT_SECONDS:
            break
        seconds_total += float(number) * _UNIT_SECONDS[unit]
        position = match.end()
    else:
        if position == len(text) and position > 0:
            return dt.timedelta(seconds=seconds_total)
    raise ConfigurationError(f"{name}: {text!r} is not a duration; use 'HH:MM:SS' or e.g. '15min'")


def format_duration(value: dt.timedelta) -> str:
    """Format a timedelta as ``"HH:MM:SS"``, the inverse of :func:`parse_duration`."""
    total = int(value.total_seconds())
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
