"""Ready-made scenarios shipped with the package.

>>> from elvis.presets import list_presets, load_preset
>>> "office" in list_presets()
True
"""

from __future__ import annotations

from importlib import resources

from core.errors import ConfigurationError
from elvis.scenario import Scenario


def list_presets() -> list[str]:
    """Names of all bundled scenarios."""
    return sorted(
        entry.name.removesuffix(".yaml")
        for entry in resources.files(__package__).iterdir()
        if entry.name.endswith(".yaml")
    )


def load_preset(name: str) -> Scenario:
    """Load a bundled scenario by name; see :func:`list_presets`."""
    resource = resources.files(__package__) / f"{name}.yaml"
    if not resource.is_file():
        raise ConfigurationError(f"unknown preset {name!r}; available: {list_presets()}")
    with resources.as_file(resource) as path:
        return Scenario.from_yaml(path)


__all__ = ["list_presets", "load_preset"]
