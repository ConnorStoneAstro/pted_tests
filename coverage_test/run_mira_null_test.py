from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coverage_test.gaussian_posteriors import simulate_posteriors
from coverage_test.run_bootstrap_coverage_test import bootstrap_posteriors, mira_score


def summarize_null_scores(scores: np.ndarray, bootstrap_scores: np.ndarray) -> dict:
    for name, values in (("Null scores", scores), ("Bootstrap scores", bootstrap_scores)):
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a finite one-dimensional array with at least two values")
    bootstrap_std = float(bootstrap_scores.std(ddof=1))
    if bootstrap_std <= 0:
        raise ValueError("Bootstrap standard deviation must be positive for null calibration")
    empirical_std = float(scores.std(ddof=1))
    z_scores = (scores - 2 / 3) / bootstrap_std
    return {
        "n_repeats": len(scores),
        "n_bootstraps": len(bootstrap_scores),
        "mira_threshold": 2 / 3,
        "null_mean": float(scores.mean()),
        "null_bias_from_threshold": float(scores.mean() - 2 / 3),
        "null_std": empirical_std,
        "bootstrap_mean": float(bootstrap_scores.mean()),
        "bootstrap_std": bootstrap_std,
        "null_to_bootstrap_std_ratio": empirical_std / bootstrap_std,
        "null_to_bootstrap_variance_ratio": (empirical_std / bootstrap_std) ** 2,
        "z_mean": float(z_scores.mean()),
        "z_std": float(z_scores.std(ddof=1)),
        "coverage": [
            {
                "std_multiplier": k,
                "observed_count": int(np.count_nonzero(np.abs(z_scores) <= k)),
                "observed_fraction": float(np.mean(np.abs(z_scores) <= k)),
                "gaussian_expected_fraction": float(norm.cdf(k) - norm.cdf(-k)),
            }
            for k in (1, 2, 3)
        ],
    }


def run_null_test(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    simulation_stream, bootstrap_stream, mira_stream = np.random.SeedSequence(args.seed).spawn(3)
    simulation_rng = np.random.default_rng(simulation_stream)
    bootstrap_rng = np.random.default_rng(bootstrap_stream)
    mira_rng = np.random.default_rng(mira_stream)
    seeds = mira_rng.integers(
        0, np.iinfo(np.int64).max, size=1 + args.n_bootstraps + args.n_repeats
    )
    config = vars(args).copy()
    config.update(
        sigma=1.0,
        mira_threshold=2 / 3,
        bootstrap_std_ddof=1,
        reference_independent_of_reruns=True,
        mira_randomness="fresh reproducible seed per evaluation",
        regeneration="ground truths, covariances, observations, and posterior samples",
    )
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    np.save(output_dir / "mira_seeds.npy", seeds)
    names = ("ground_truth", "data_values", "data_cov", "posterior_samples")
    reference = simulate_posteriors(simulation_rng, args.n_sims, args.n_posterior_samples)
    for name, values in zip(names, reference):
        np.save(output_dir / f"reference_{name}.npy", values)
    truth = torch.from_numpy(reference[0])
    reference_score = mira_score(truth, reference[3], args.mira_runs, int(seeds[0]))
    np.save(output_dir / "reference_score_mira.npy", reference_score)
    print(f"Independent reference MIRA score: {reference_score:.6f}", flush=True)
    bootstrap_scores = np.empty(args.n_bootstraps, dtype=np.float64)
    for i in range(args.n_bootstraps):
        bootstrap_scores[i] = mira_score(
            truth,
            bootstrap_posteriors(reference[3], bootstrap_rng),
            args.mira_runs,
            int(seeds[i + 1]),
        )
        if (i + 1) % 32 == 0 or i + 1 == args.n_bootstraps:
            print(f"Reference bootstrap {i + 1}/{args.n_bootstraps}", flush=True)
    np.save(output_dir / "bootstrap_scores_mira.npy", bootstrap_scores)
    scores = np.empty(args.n_repeats, dtype=np.float64)
    systems = [np.empty((args.n_repeats, *values.shape), dtype=values.dtype) for values in reference]
    for i in range(args.n_repeats):
        system = simulate_posteriors(simulation_rng, args.n_sims, args.n_posterior_samples)
        for values, storage in zip(system, systems):
            storage[i] = values
        scores[i] = mira_score(
            torch.from_numpy(system[0]),
            system[3],
            args.mira_runs,
            int(seeds[1 + args.n_bootstraps + i]),
        )
        if (i + 1) % 32 == 0 or i + 1 == args.n_repeats:
            print(f"Independent null system {i + 1}/{args.n_repeats}", flush=True)
    np.save(output_dir / "scores_mira.npy", scores)
    for name, values in zip(names, systems):
        np.save(output_dir / f"{name}.npy", values)
    summary = summarize_null_scores(scores, bootstrap_scores)
    summary["reference_score"] = reference_score
    summary["bootstrap_bias_from_reference"] = float(bootstrap_scores.mean() - reference_score)
    np.save(output_dir / "bootstrap_std_mira.npy", summary["bootstrap_std"])
    np.save(output_dir / "z_scores_mira.npy", (scores - 2 / 3) / summary["bootstrap_std"])
    (output_dir / "null_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Wrote MIRA null results to: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare independent MIRA null scores with reference bootstrap uncertainty")
    parser.add_argument("--output-dir", default="coverage_test/results/mira_null_test")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration and exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sims", type=int, default=64)
    parser.add_argument("--n-posterior-samples", type=int, default=128)
    parser.add_argument("--n-repeats", type=int, default=256)
    parser.add_argument("--n-bootstraps", type=int, default=256)
    parser.add_argument("--mira-runs", type=int, default=512)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    for name in ("n_sims", "n_posterior_samples", "n_repeats", "n_bootstraps", "mira_runs"):
        if getattr(args, name) < 2:
            parser.error(f"--{name.replace('_', '-')} must be at least 2")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be between 0 and 2**63 - 1")
    if args.dry_run:
        config = vars(args).copy()
        config.update(sigma=1.0, mira_evaluations=1 + args.n_bootstraps + args.n_repeats)
        print("Dry run configuration")
        print(json.dumps(config, indent=2))
        return
    run_null_test(args)


if __name__ == "__main__":
    main()
