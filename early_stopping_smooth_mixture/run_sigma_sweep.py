from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import torch
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.metrics import metric_sweep
from early_stopping.two_moons import sample_two_moons, two_moons_density


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a smooth sigma sweep for a two-moons Gaussian mixture model "
            "using fixed base samples scaled by sigma"
        )
    )
    parser.add_argument(
        "--output-dir",
        default="early_stopping_smooth_mixture/results/two_moons_sigma_sweep",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric-seed", type=int, default=11)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-generated", type=int, default=1024)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--n-sigmas", type=int, default=120)
    parser.add_argument(
        "--sigma-schedule",
        choices=["log", "linear"],
        default="log",
        help="Spacing strategy for sigma values between sigma-max and sigma-min",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save generated samples for each sigma (can be large)",
    )
    return parser


def _build_sigma_values(args: argparse.Namespace) -> np.ndarray:
    if args.n_sigmas <= 1:
        raise ValueError("--n-sigmas must be >= 2")
    if args.sigma_max <= 0.0 or args.sigma_min <= 0.0:
        raise ValueError("--sigma-max and --sigma-min must be positive")
    if args.sigma_max <= args.sigma_min:
        raise ValueError("--sigma-max must be larger than --sigma-min")

    if args.sigma_schedule == "log":
        values = np.geomspace(args.sigma_max, args.sigma_min, args.n_sigmas)
    else:
        values = np.linspace(args.sigma_max, args.sigma_min, args.n_sigmas)
    return values.astype(np.float32)


def _save_reference_density(output_dir: Path, noise: float) -> None:
    xx, yy = np.meshgrid(
        np.linspace(-1.8, 2.8, 220),
        np.linspace(-1.4, 1.8, 220),
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    density = two_moons_density(grid, noise=noise, quadrature_points=512).reshape(xx.shape)
    np.savez_compressed(
        output_dir / "reference_density.npz",
        xx=xx.astype(np.float32),
        yy=yy.astype(np.float32),
        density=density.astype(np.float32),
    )


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "sigma_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    sigma_values = _build_sigma_values(args)
    np.save(output_dir / "sigma_values.npy", sigma_values)

    rng = np.random.default_rng(args.seed)

    train_data = sample_two_moons(args.n_train, rng=rng, noise=args.noise)
    np.save(output_dir / "train_data.npy", train_data)
    _save_reference_density(output_dir=output_dir, noise=args.noise)

    component_indices = rng.integers(0, len(train_data), size=args.n_generated)
    base_noise = rng.normal(size=(args.n_generated, 2)).astype(np.float32)
    centers = train_data[component_indices]

    np.save(output_dir / "component_indices.npy", component_indices.astype(np.int32))
    np.save(output_dir / "base_noise.npy", base_noise)
    np.save(output_dir / "base_centers.npy", centers.astype(np.float32))

    metrics = metric_sweep()
    metric_names = list(metrics.keys())

    records: list[dict[str, float | int | str]] = []

    for sigma_idx, sigma in enumerate(sigma_values):
        generated = centers + float(sigma) * base_noise

        if args.save_samples:
            np.save(samples_dir / f"sigma_{sigma_idx:04d}.npy", generated.astype(np.float32))

        for metric_offset, metric_name in enumerate(metric_names):
            metric_fn = metrics[metric_name]
            metric_rng = np.random.default_rng(args.metric_seed + metric_offset)
            torch.manual_seed(args.metric_seed + metric_offset)
            score = metric_fn(
                train_data,
                generated,
                permutations=args.permutations,
                rng=metric_rng,
            )
            records.append(
                {
                    "sigma_idx": sigma_idx,
                    "sigma": float(sigma),
                    "method": metric_name,
                    "score": float(score),
                }
            )

        print(
            f"sigma_idx={sigma_idx:04d} "
            f"sigma={float(sigma):.6f} "
            f"generated_shape={generated.shape}"
        )

    with (output_dir / "sweep_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sigma_idx", "sigma", "method", "score"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved smooth mixture sweep artifacts to: {output_dir}")
    print(
        "Next step: run `python early_stopping_smooth_mixture/plot_sigma_sweep.py "
        f"--output-dir {output_dir}` to build plots and GIF."
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
