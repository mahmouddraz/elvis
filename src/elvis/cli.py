"""Command line interface: ``elvis presets`` and ``elvis run``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from core.errors import CoreError
from elvis.presets import list_presets, load_preset
from elvis.scenario import Scenario
from elvis.simulation import simulate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="elvis", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("presets", help="list bundled scenarios")

    run = commands.add_parser("run", help="simulate a preset or scenario file")
    run.add_argument("scenario", help="preset name or path to a scenario YAML file")
    run.add_argument("--start", required=True, help='first step, e.g. "2024-01-01"')
    run.add_argument("--end", required=True, help='last step, e.g. "2024-01-31 23:45"')
    run.add_argument("--resolution", default="00:15:00", help="step length (default 15 min)")
    run.add_argument("--seed", type=int, help="seed for reproducible sampling")
    run.add_argument("--csv", type=Path, help="write the power time series to this file")

    args = parser.parse_args(argv)
    try:
        if args.command == "presets":
            print("\n".join(list_presets()))
            return 0
        source = Path(args.scenario)
        scenario = (
            Scenario.from_yaml(source)
            if source.suffix in (".yaml", ".yml")
            else (load_preset(args.scenario))
        )
        result = simulate(scenario, args.start, args.end, args.resolution, seed=args.seed)
    except CoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.csv:
        result.to_dataframe().to_csv(args.csv)
    print(json.dumps(result.summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
