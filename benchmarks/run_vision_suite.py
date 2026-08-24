from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.metrics import metric_sweep, pted_two_sample_pvalue_onetail
from benchmarks.plots import plot_method_sweep
from benchmarks.runners.vision import run_vision_sweep
from utils import _grab_config, _annotate
from benchmarks.run_gaussian_suite import _read_records_csv


def _write_records_csv(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    keys: list[str] = []
    seen: set[str] = set()
    for row in records:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _plot_vision_outputs(records: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted({str(record["dataset"]) for record in records})
    deviations = sorted({str(record["deviation"]) for record in records})
    deviation_titles = {
        "pair_blend": "$y \\sim (1 - \\frac{{\\rm S}}{2})y_1 + \\frac{{\\rm S}}{2} y_2,~~ y_1,y_2\\sim \\mathcal{D}$",
        "class_drop": "$y$ from $\\mathcal{D}$ where class 0 has $(1-{\\rm S})$ weight",
        "white_noise": "$y$ from $\\mathcal{D} + \\mathcal{N}(0, (\\frac{{\\rm S}}{3})^2)$",
    }
    dataset_names = {
        "mnist": "MNIST",
        "cifar10": "CIFAR-10",
        "gaussian2x2": "Gaussian 2x2",
    }
    for dataset in datasets:
        print(f"Plotting vision outputs for dataset: {dataset}")
        for deviation in deviations:
            print(f"Plotting vision outputs for deviation: {deviation}")
            subset = [
                record
                for record in records
                if str(record["deviation"]) == deviation and str(record["dataset"]) == dataset
            ]
            plot_method_sweep(
                subset,
                output_path=out_dir / f"vision_{dataset}_{deviation}_score.png",
                title=f"Vision test {dataset_names[dataset]}: {deviation_titles[deviation]}",
                x_key="severity",
                y_key="score",
                ylabel="p-value",
                new_yscale=["fld", "fid"],
            )


def run_suite(config: dict[str, Any], dry_run: bool = False) -> None:
    if dry_run:
        print("Dry run configuration")
        print(json.dumps(config, indent=2))
        return

    output_dir = Path(config["output_dir"]) / "vision"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running suite: vision")
    metrics = metric_sweep()
    metrics["pted"] = pted_two_sample_pvalue_onetail
    records = run_vision_sweep(
        datasets=config["datasets"],
        deviations=config["deviations"],
        severities=config["severities"],
        n_samples=config["n_samples"],
        seeds=config["seeds"],
        permutations=config["permutations"],
        metrics=metrics,
    )
    _annotate(records, suite="vision")

    _write_records_csv(records, output_dir / "vision_suite_records.csv")
    _plot_vision_outputs(records, output_dir)
    print(f"Wrote records: {output_dir / 'vision_suite_records.csv'}")
    for metric in metric_sweep().keys():
        subset = [record for record in records if record["method"] == metric]
        print(
            f"Metric: {metric} total runtime: {sum(record['runtime'] for record in subset):.4f} seconds"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run vision benchmark suite from python configuration file"
    )
    parser.add_argument(
        "--config",
        default="benchmarks/configs/vision_suite.py",
        help="Python configuration file path",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip experiments and regenerate plots from an existing records CSV",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print merged configuration and exit",
    )
    parser.add_argument(
        "--records-csv",
        default=None,
        help=(
            "Path to an existing vision records CSV. "
            "Defaults to <output_dir>/vision/vision_suite_records.csv"
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = _grab_config(args.config)
    if args.plot_only:
        output_dir = Path(config["output_dir"])
        vision_out = output_dir / "vision"
        default_csv = vision_out / "vision_suite_records.csv"
        csv_path = Path(args.records_csv) if args.records_csv else default_csv
        records = _read_records_csv(csv_path)
        _plot_vision_outputs(records, vision_out)
        print(f"Regenerated plots from: {csv_path}")
        return
    run_suite(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
