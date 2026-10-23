from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from mira_score import mira
from pted import pted_coverage_test

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coverage_test.gaussian_posteriors import simulate_posteriors


def build_sigma_values(sigma_min: float, sigma_max: float, sigma_step: float) -> np.ndarray:
    if not all(np.isfinite(value) for value in (sigma_min, sigma_max, sigma_step)):
        raise ValueError("Sigma limits and step must be finite")
    if sigma_min <= 0 or sigma_max < sigma_min or sigma_step <= 0:
        raise ValueError("Require 0 < sigma_min <= sigma_max and sigma_step > 0")
    intervals = (sigma_max - sigma_min) / sigma_step
    if not np.isfinite(intervals) or not np.isclose(intervals, round(intervals), rtol=0, atol=1e-9):
        raise ValueError("The sigma range must be an integer multiple of sigma_step")
    return np.linspace(sigma_min, sigma_max, round(intervals) + 1, dtype=np.float64)


def bootstrap_posteriors(samples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_samples, n_sims, _ = samples.shape
    indices = rng.integers(n_samples, size=(n_samples, n_sims))
    return samples[indices, np.arange(n_sims)[None, :], :]


def mira_score(
    truth: torch.Tensor, samples: np.ndarray, num_runs: int, seed: int
) -> float:
    torch.manual_seed(seed)
    scores, _ = mira(
        truth,
        torch.from_numpy(np.moveaxis(samples, 0, 1)[None]),
        num_runs=num_runs,
        norm=True,
        disable_tqdm=True,
    )
    return scores.item()


def run_bootstrap_sweep(args: argparse.Namespace) -> None:
    sigma_values = build_sigma_values(args.sigma_min, args.sigma_max, args.sigma_step)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ground_truth, data_values, data_cov, posterior_samples = simulate_posteriors(
        rng, args.n_sims, args.n_posterior_samples
    )
    bootstrap_stream, mira_stream, pted_stream = np.random.SeedSequence(args.seed).spawn(3)
    bootstrap_rng = np.random.default_rng(bootstrap_stream)
    mira_rng = np.random.default_rng(mira_stream)
    pted_rng = np.random.default_rng(pted_stream)
    mira_seeds = mira_rng.integers(
        0, np.iinfo(np.int64).max, size=(len(sigma_values), args.n_bootstraps + 1)
    )
    truth = torch.from_numpy(ground_truth)
    pvalues_pted = np.empty(len(sigma_values), dtype=np.float64)
    scores_mira = np.empty(len(sigma_values), dtype=np.float64)
    bootstrap_scores = np.empty((len(sigma_values), args.n_bootstraps), dtype=np.float64)
    config = vars(args).copy()
    config.update(
        sigma_values=sigma_values.tolist(),
        mira_threshold=2 / 3,
        bootstrap_std_ddof=1,
        window_std_multiplier=1,
        mira_randomness="fresh reproducible seed per evaluation",
    )
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    for i, sigma in enumerate(sigma_values):
        print(f"Sigma {sigma:.1f} ({i + 1}/{len(sigma_values)}): PTED and MIRA", flush=True)
        scaled_samples = (
            posterior_samples - data_values[None, :, :]
        ) * sigma + data_values[None, :, :]
        torch.manual_seed(args.seed)
        pvalues_pted[i] = pted_coverage_test(
            truth,
            torch.from_numpy(scaled_samples),
            permutations=args.permutations,
            pit_plot=output_dir / f"pit_plot_sigma_{i:04d}.pdf",
            rng=pted_rng,
        )
        scores_mira[i] = mira_score(
            truth, scaled_samples, args.permutations, int(mira_seeds[i, 0])
        )
        for j in range(args.n_bootstraps):
            bootstrap_scores[i, j] = mira_score(
                truth,
                bootstrap_posteriors(scaled_samples, bootstrap_rng),
                args.permutations,
                int(mira_seeds[i, j + 1]),
            )
            if (j + 1) % 32 == 0 or j + 1 == args.n_bootstraps:
                print(f"  MIRA bootstrap {j + 1}/{args.n_bootstraps}", flush=True)
    artifacts = {
        "sigma_values": sigma_values,
        "pvalues_pted": pvalues_pted,
        "scores_mira": scores_mira,
        "bootstrap_scores_mira": bootstrap_scores,
        "bootstrap_std_mira": bootstrap_scores.std(axis=1, ddof=1),
        "mira_seeds": mira_seeds,
        "ground_truth": ground_truth,
        "data_values": data_values,
        "data_cov": data_cov,
        "posterior_samples": posterior_samples,
    }
    for name, values in artifacts.items():
        np.save(output_dir / f"{name}.npy", values)
    print(f"Wrote bootstrap coverage results to: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PTED and bootstrap-calibrated MIRA coverage sweep")
    parser.add_argument("--output-dir", default="coverage_test/results/bootstrap_coverage_test")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration and exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sims", type=int, default=64)
    parser.add_argument("--n-posterior-samples", type=int, default=128)
    parser.add_argument("--permutations", type=int, default=512, help="PTED permutations / MIRA runs")
    parser.add_argument("--n-bootstraps", type=int, default=256)
    parser.add_argument("--sigma-min", type=float, default=0.5)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--sigma-step", type=float, default=0.1)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    for name in ("n_sims", "n_posterior_samples", "permutations", "n_bootstraps"):
        if getattr(args, name) < 2:
            parser.error(f"--{name.replace('_', '-')} must be at least 2")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be between 0 and 2**63 - 1")
    try:
        sigma_values = build_sigma_values(args.sigma_min, args.sigma_max, args.sigma_step)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        config = vars(args).copy()
        config["sigma_values"] = sigma_values.tolist()
        print("Dry run configuration")
        print(json.dumps(config, indent=2))
        return
    run_bootstrap_sweep(args)


if __name__ == "__main__":
    main()
