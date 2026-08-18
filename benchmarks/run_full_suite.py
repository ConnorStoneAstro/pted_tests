from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmarks.run_gaussian_suite as gaussian_suite
import benchmarks.run_vision_suite as vision_suite


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full benchmark suite from YAML")
    parser.add_argument(
        "--config",
        default="benchmarks/configs/full_suite.yaml",
        help="YAML configuration file path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print merged configuration and exit",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config_path = Path(args.config)
    gaussian_config = gaussian_suite._load_config(config_path)
    vision_config = vision_suite._load_config(config_path)

    if args.dry_run:
        combined = dict(gaussian_config)
        combined["vision"] = vision_config["vision"]
        print("Dry run configuration")
        import json

        print(json.dumps(combined, indent=2))
        return

    gaussian_suite.run_suite(gaussian_config, dry_run=False)
    vision_suite.run_suite(vision_config, dry_run=False)


if __name__ == "__main__":
    main()
