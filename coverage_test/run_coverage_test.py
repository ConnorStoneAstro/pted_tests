from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pted import pted_coverage_test, utils
import torch
from mira_score import mira
import scipy.stats

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.metrics import metric_sweep


def sample_data_covariance(rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(0.5, 2.0, size=2)
    cov = rng.uniform(-1, 1) * np.prod(sigma)
    return np.array([[sigma[0] ** 2, cov], [cov, sigma[1] ** 2]]).astype(np.float64)


def build_sigma_values(sigma_min: float, sigma_max: float, n_sigmas: int) -> np.ndarray:
    return np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_sigmas, dtype=np.float64)


def run_coverage_sweep(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    true_param_cov = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float64)

    ground_truth = rng.multivariate_normal(mean=np.zeros(2), cov=true_param_cov, size=args.n_sims)

    data_values = []
    data_cov = []
    for sim_idx in range(args.n_sims):
        data_cov.append(sample_data_covariance(rng))
        data_values.append(
            rng.multivariate_normal(mean=ground_truth[sim_idx], cov=data_cov[sim_idx])
        )
    data_cov = np.stack(data_cov, axis=0)
    data_values = np.stack(data_values, axis=0)  # (Nsim, 2)

    posterior_samples = np.stack(
        [
            rng.multivariate_normal(mean=dv, cov=dc, size=args.n_posterior_samples)
            for dv, dc in zip(data_values, data_cov)
        ],
        axis=1,
    )  # (Nsamp, Nsim, 2)

    sigma_values = build_sigma_values(args.sigma_min, args.sigma_max, args.n_sigmas)
    pvalues_pted = np.zeros(len(sigma_values), dtype=np.float64)
    pvalues_mira = np.zeros(len(sigma_values), dtype=np.float64)
    pvalues_hdp = np.zeros(len(sigma_values), dtype=np.float64)
    pvalues_mmd = np.zeros(len(sigma_values), dtype=np.float64)
    pvalues_ks = np.zeros(len(sigma_values), dtype=np.float64)
    MMD = metric_sweep()["mmd"]
    for i, sigma in enumerate(sigma_values):
        scaled_posterior_samples = (
            posterior_samples - data_values[None, :, :]
        ) * sigma + data_values[None, :, :]
        # PTED
        torch.manual_seed(42)
        pvalues_pted[i] = pted_coverage_test(
            torch.tensor(ground_truth),
            torch.tensor(scaled_posterior_samples),
            permutations=args.permutations,
            pit_plot=output_dir / f"pit_plot_sigma_{i:04d}.pdf",
        )
        # MIRA
        torch.manual_seed(42)
        pvalues_mira[i] = mira(
            torch.tensor(ground_truth),
            torch.tensor(np.moveaxis(scaled_posterior_samples[None], 1, 2)),
            num_runs=args.permutations,
            norm=True,
        )[0]
        # MMD
        pvals_mmd = np.array(
            [
                MMD(g.reshape(1, -1), p, permutations=args.permutations, rng=rng)
                for g, p in zip(
                    torch.tensor(ground_truth),
                    torch.tensor(np.moveaxis(scaled_posterior_samples, 1, 0)),
                )
            ]
        )
        pvalues_mmd[i] = utils.two_tailed_p(-2 * np.sum(np.log(pvals_mmd + 1e-10)), 2 * args.n_sims)
        # HDP
        chi2_hdp = 0
        for gt, dv, dc, sps in zip(
            ground_truth, data_values, data_cov, np.moveaxis(scaled_posterior_samples, 0, 1)
        ):
            gt_density = scipy.stats.multivariate_normal.pdf(gt, mean=dv, cov=sigma * dc)
            sps_density = np.stack(
                [scipy.stats.multivariate_normal.pdf(sps_, mean=dv, cov=sigma * dc) for sps_ in sps]
            )
            chi2_hdp += -2 * np.log(
                (np.sum(sps_density >= gt_density[None], axis=0) + 1) / (sps.shape[0] + 1)
            )
        pvalues_hdp[i] = utils.two_tailed_p(chi2_hdp, 2 * args.n_sims)
        # KS
        pvals_ks = np.array(
            [
                scipy.stats.kstest(g, s)[1]
                for g, s in zip(
                    np.sum(ground_truth, axis=1),
                    np.moveaxis(np.sum(scaled_posterior_samples, axis=-1), 1, 0),
                )
            ]
        )
        pvalues_ks[i] = utils.two_tailed_p(-2 * np.sum(np.log(pvals_ks + 1e-10)), 2 * args.n_sims)
    np.save(output_dir / "pvalues_pted.npy", pvalues_pted)
    np.save(output_dir / "pvalues_mira.npy", pvalues_mira)
    np.save(output_dir / "pvalues_hdp.npy", pvalues_hdp)
    np.save(output_dir / "pvalues_mmd.npy", pvalues_mmd)
    np.save(output_dir / "pvalues_ks.npy", pvalues_ks)
    np.save(output_dir / "sigma_values.npy", sigma_values)
    np.save(output_dir / "ground_truth.npy", ground_truth)
    np.save(output_dir / "data_values.npy", data_values)
    np.save(output_dir / "data_cov.npy", data_cov)
    np.save(output_dir / "posterior_samples.npy", posterior_samples)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Gaussian coverage sweep experiment")
    parser.add_argument("--output-dir", default="coverage_test/results/coverage_test")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration and exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sims", type=int, default=64)
    parser.add_argument("--n-posterior-samples", type=int, default=128)
    parser.add_argument("--permutations", type=int, default=512)
    parser.add_argument("--sigma-min", type=float, default=0.5)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--n-sigmas", type=int, default=64)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.dry_run:
        config = vars(args).copy()
        config["sigma_values"] = build_sigma_values(
            args.sigma_min,
            args.sigma_max,
            args.n_sigmas,
        ).tolist()
        print("Dry run configuration")
        print(json.dumps(config, indent=2))
        return
    run_coverage_sweep(args)


if __name__ == "__main__":
    main()
