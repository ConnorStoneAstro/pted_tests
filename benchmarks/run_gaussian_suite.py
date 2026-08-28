from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.plots import plot_method_sweep
from benchmarks.runners.gaussian import run_gaussian_sweep
from benchmarks.utils import _grab_config, _annotate
from benchmarks.metrics import metric_sweep, pted_two_sample_pvalue_onetail


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


def _read_records_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Records CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        records = [dict(row) for row in reader]

    if not records:
        raise ValueError(f"Records CSV is empty: {path}")

    return records


def _plot_gaussian_outputs(records: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    deviations = sorted({str(record["deviation"]) for record in records})
    deviation_titles = {
        "mean_shift": "$y \\sim \\mathcal{N}({\\rm S}, 1)$",
        "scale_shift": "$y \\sim \\mathcal{N}(0, (1 + {\\rm S})^2)$",
        "bimodal": "$y \\sim 0.5 \\mathcal{N}(-{\\rm S}, 1) + 0.5 \\mathcal{N}({\\rm S}, 1)$",
        "skew": "$y \\sim \\text{SkewNormal}(0, 1, {\\rm S})$",
        "contamination": "$y \\sim (1 - {\\rm S}) \\mathcal{N}(0, 1) + {\\rm S} \\mathcal{N}(4, 1)$",
    }
    for deviation in deviations:
        print(f"Plotting gaussian outputs for deviation: {deviation}")
        subset = [record for record in records if str(record["deviation"]) == deviation]
        plot_method_sweep(
            subset,
            output_path=out_dir / f"gaussian_1d_{deviation}_score.pdf",
            title=f"Gaussian 1D test: $x \\sim \\mathcal{{N}}(0, 1)$, {deviation_titles[deviation]}",
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

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running suite: gaussian_1d")
    metrics = metric_sweep()
    metrics["pted"] = pted_two_sample_pvalue_onetail
    records = run_gaussian_sweep(
        deviations=config["deviations"],
        severities=config["severities"],
        n_samples=config["n_samples"],
        permutations=config["permutations"],
        seeds=config["seeds"],
        metrics=metrics,
    )
    _annotate(records, suite="gaussian_1d")

    gaussian_out = output_dir / "gaussian_1d"
    _write_records_csv(records, gaussian_out / "gaussian_1d_records.csv")
    _plot_gaussian_outputs(records, gaussian_out)
    print(f"Wrote records: {gaussian_out / 'gaussian_1d_records.csv'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gaussian benchmark suite")
    parser.add_argument(
        "--config",
        default="benchmarks/configs/gaussian_suite.py",
        help="Python configuration file path which contains a CONFIG dictionary",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print merged configuration and exit",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip experiments and regenerate plots from an existing records CSV",
    )
    parser.add_argument(
        "--records-csv",
        default=None,
        help=(
            "Path to an existing gaussian records CSV. "
            "Defaults to <output_dir>/gaussian_1d/gaussian_1d_records.csv"
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = _grab_config(args.config)

    if args.dry_run:
        print("Dry run configuration")
        print(json.dumps(config, indent=2))
        if args.plot_only:
            default_csv = Path(config["output_dir"]) / "gaussian_1d" / "gaussian_1d_records.csv"
            csv_path = Path(args.records_csv) if args.records_csv else default_csv
            print(f"Plot-only CSV path: {csv_path}")
        return

    if args.plot_only:
        output_dir = Path(config["output_dir"])
        gaussian_out = output_dir / "gaussian_1d"
        default_csv = gaussian_out / "gaussian_1d_records.csv"
        csv_path = Path(args.records_csv) if args.records_csv else default_csv
        records = _read_records_csv(csv_path)
        _plot_gaussian_outputs(records, gaussian_out)
        print(f"Regenerated plots from: {csv_path}")
        return

    run_suite(config, dry_run=False)


if __name__ == "__main__":
    main()
