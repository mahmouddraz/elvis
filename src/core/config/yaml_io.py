"""Reading and writing human-friendly YAML input files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from core.errors import ConfigurationError


def load_yaml(path: str | Path) -> Any:
    """Parse a YAML file with the safe loader."""
    try:
        with Path(path).open(encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"not valid YAML: {error}", source=str(path)) from error


def dump_yaml(data: Any, path: str | Path) -> None:
    """Write ``data`` with numeric lists inline and integral floats as integers."""
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.dump(data, file, Dumper=_CompactDumper, sort_keys=False, width=100)


class _CompactDumper(yaml.SafeDumper):
    """Lists of numbers (profiles, matrices) inline, everything else in block style."""


def _represent_list(dumper: yaml.SafeDumper, data: list[Any]) -> yaml.Node:
    inline = all(isinstance(x, (int, float)) for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=inline)


def _represent_float(dumper: yaml.SafeDumper, data: float) -> yaml.Node:
    if data.is_integer() and abs(data) < 1e15:
        return dumper.represent_int(int(data))
    return dumper.represent_float(data)


_CompactDumper.add_representer(list, _represent_list)
_CompactDumper.add_representer(float, _represent_float)
