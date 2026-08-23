from __future__ import annotations

import argparse

from .build import build_dataset
from .publish import publish_dataset
from .validate import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(prog="dataset")
    parser.add_argument("command", choices=("build", "validate", "publish"))
    parser.add_argument("--config", default="dataset.toml")
    args = parser.parse_args()

    if args.command == "build":
        result = build_dataset(args.config)
        print(f"built {result.row_count} rows -> {result.output_path}")
    elif args.command == "validate":
        report = validate_dataset(args.config)
        print(f"validated {report.row_count} rows -> {report.output_path}")
    else:
        print(publish_dataset(args.config))
