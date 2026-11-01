"""Allocation policies decide the service rate of every occupied server in each step.

Input files name a policy by :attr:`Policy.name`, optionally with parameters::

    policy: earliest_departure_first
    policy: {policy: fair_share, block: "00:15:00"}

Custom policies subclass :class:`Policy` and are added with :func:`register_policy`. Most
limit-aware policies only need to choose a priority order and call
:func:`allocate_in_priority_order`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.allocation.base import Allocation, Policy, StepContext, allocate_in_priority_order
from core.allocation.policies import (
    EarliestDepartureFirst,
    FairShare,
    FirstComeFirstServed,
    Unconstrained,
)
from core.errors import ConfigurationError

POLICIES: dict[str, type[Policy]] = {
    cls.name: cls
    for cls in (Unconstrained, FirstComeFirstServed, EarliestDepartureFirst, FairShare)
}


def register_policy(cls: type[Policy]) -> type[Policy]:
    """Make a policy class usable from input files; usable as a decorator."""
    POLICIES[cls.name] = cls
    return cls


def policy_from_config(raw: str | Mapping[str, Any]) -> Policy:
    """Create a fresh policy from a name or ``{policy: name, **parameters}``."""
    params = {"policy": raw} if isinstance(raw, str) else dict(raw)
    name = params.pop("policy", None)
    if name not in POLICIES:
        raise ConfigurationError(f"policy must be one of {sorted(POLICIES)}, got {name!r}")
    try:
        return POLICIES[name](**params)
    except TypeError as error:
        raise ConfigurationError(f"invalid parameters for policy {name!r}: {error}") from error


__all__ = [
    "POLICIES",
    "Allocation",
    "EarliestDepartureFirst",
    "FairShare",
    "FirstComeFirstServed",
    "Policy",
    "StepContext",
    "Unconstrained",
    "allocate_in_priority_order",
    "policy_from_config",
    "register_policy",
]
