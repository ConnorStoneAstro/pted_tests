from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.datasets.vision import load_vision_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load MNIST and CIFAR10 from a data directory")
    parser.add_argument(
        "--data-root",
        default="benchmarks/data",
        help="Directory containing the MNIST and CIFAR10 data",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the datasets into the data directory if needed",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["mnist", "cifar10"],
        help="Datasets to load (default: mnist cifar10)",
    )
    return parser


def _summarize_dataset(dataset: str, data_root: Path, download: bool) -> None:
    images, labels = load_vision_dataset(dataset=dataset, data_root=data_root, download=download)
    label_values, label_counts = np.unique(labels, return_counts=True)
    counts = ", ".join(
        f"{int(label)}:{int(count)}" for label, count in zip(label_values, label_counts)
    )
    print(f"{dataset}: images={images.shape}, labels={labels.shape}, classes={len(label_values)}")
    print(f"  label_counts: {counts}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    data_root = Path(args.data_root)

    for dataset in args.datasets:
        _summarize_dataset(dataset, data_root=data_root, download=args.download)


if __name__ == "__main__":
    main()
