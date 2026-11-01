"""Exception hierarchy.

Every error raised on purpose derives from :class:`CoreError`, so callers can catch the
package's failures without also swallowing unrelated bugs.
"""

from __future__ import annotations


class CoreError(Exception):
    """Base class for all errors raised by ``core`` and the applications built on it."""


class ConfigurationError(CoreError, ValueError):
    """A model, network or scenario definition is invalid.

    Collects every problem found during validation, so a user can fix an input file in one
    pass instead of one error at a time.
    """

    def __init__(self, problems: str | list[str], *, source: str | None = None) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        self.source = source
        prefix = f"{source}: " if source else ""
        if len(self.problems) == 1:
            message = prefix + self.problems[0]
        else:
            message = (
                prefix
                + "invalid configuration:\n"
                + "\n".join(f"  - {problem}" for problem in self.problems)
            )
        super().__init__(message)


class SimulationError(CoreError, RuntimeError):
    """The simulation reached a state that violates a physical or logical invariant."""
