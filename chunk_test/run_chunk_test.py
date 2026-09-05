from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import process_time
from typing import Any

import numpy as np
import torch
from pted import pted

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.datasets.vision import generate_vision_problem, load_vision_dataset
from benchmarks.metrics import DEVICE, _prepare_samples
from benchmarks.utils import _grab_config
from chunk_test.plot_chunk_test import plot_chunk_runtime, plot_chunk_sweep

DATASET_NAMES = {
    "mnist": "MNIST",
    "cifar10": "CIFAR-10",
    "gaussian2x2": "Gaussian 2x2",
}
DEVIATION_TITLES = {
    "pair_blend": "$y \\sim (1 - \\frac{{\\rm S}}{2})y_1 + \\frac{{\\rm S}}{2} y_2,~~ y_1,y_2\\sim \\mathcal{D}$",
    "class_drop": "$y$ from $\\mathcal{D}$ where class 0 has $(1-{\\rm S})$ weight",
    "white_noise": "$y$ from $\\mathcal{D} + \\mathcal{N}(0, (\\frac{{\\rm S}}{3})^2)$",
}


def _write_records_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _read_records_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_chunk_sweep(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_name = config["dataset"]
    deviation = config["deviation"]
    n_samples = int(config["n_samples"])
    permutations = int(config["permutations"])
    two_tailed = bool(config.get("two_tailed", False))

    dataset = load_vision_dataset(dataset_name)
    records: list[dict[str, Any]] = []

    for severity in config["severities"]:
        print(f"Severity: {severity:.3f}")
        for seed in config["seeds"]:
            rng = np.random.default_rng(seed)
            problem = generate_vision_problem(
                dataset=dataset,
                severity=severity,
                deviation=deviation,
                n_samples=n_samples,
                rng=rng,
            )
            x, y = _prepare_samples(problem.x, problem.y)
            x_t = torch.tensor(x, device=DEVICE)
            y_t = torch.tensor(y, device=DEVICE)

            for chunk_size in config["chunk_sizes"]:
                torch.manual_seed(seed)
                start = process_time()
                value = float(
                    pted(
                        x_t,
                        y_t,
                        permutations=permutations,
                        chunk_size=int(chunk_size),
                        two_tailed=two_tailed,
                    )
                )
                runtime = process_time() - start
                records.append(
                    {
                        "method": "pted",
                        "dataset": dataset_name,
                        "deviation": deviation,
                        "severity": float(severity),
                        "seed": int(seed),
                        "chunk_size": int(chunk_size),
                        "n_samples": n_samples,
                        "score": value,
                        "runtime": runtime,
                    }
                )
    return records


def _plot_outputs(records: list[dict[str, Any]], config: dict[str, Any], out_dir: Path) -> None:
    dataset_name = str(records[0]["dataset"])
    deviation = str(records[0]["deviation"])
    title = (
        f"PTED chunk size sweep {DATASET_NAMES.get(dataset_name, dataset_name)}: "
        f"{DEVIATION_TITLES.get(deviation, deviation)}"
    )
    plot_chunk_sweep(
        records,
        output_path=out_dir / f"chunk_{dataset_name}_{deviation}_score.pdf",
        title=title,
    )
    plot_chunk_runtime(
        records,
        output_path=out_dir / f"chunk_{dataset_name}_{deviation}_runtime.pdf",
        title="PTED runtime vs chunk size",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep the PTED chunk_size parameter on a vision benchmark problem"
    )
    parser.add_argument("--config", default="chunk_test/config.py", help="Configuration file path")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration and exit")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip experiments and regenerate plots from an existing records CSV",
    )
    parser.add_argument("--records-csv", default=None, help="Path to an existing records CSV")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = _grab_config(args.config)
    output_dir = Path(config["output_dir"])

    if args.dry_run:
        print("Dry run configuration")
        print(json.dumps(config, indent=2))
        return

    csv_path = Path(args.records_csv) if args.records_csv else output_dir / "chunk_test_records.csv"

    if args.plot_only:
        records = _read_records_csv(csv_path)
    else:
        records = run_chunk_sweep(config)
        _write_records_csv(records, csv_path)
        with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Wrote records: {csv_path}")

    _plot_outputs(records, config, output_dir)
    print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()
