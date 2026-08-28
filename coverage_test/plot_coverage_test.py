from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from pted import pted_coverage_test

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Gaussian coverage-sweep outputs")
    parser.add_argument("--output-dir", default="coverage_test/results/coverage_test")
    parser.add_argument("--gif-name", default="coverage_sweep_animation.gif")
    parser.add_argument("--gif-fps", type=float, default=6.0)
    parser.add_argument("--gif-loop", type=int, default=0)
    return parser


def _load_run_artifacts(output_dir: Path) -> dict[str, Any]:
    sigma_values = np.load(output_dir / "sigma_values.npy")
    pvalues = np.load(output_dir / "pvalues_pted.npy")
    ground_truth = np.load(output_dir / "ground_truth.npy")
    data_values = np.load(output_dir / "data_values.npy")
    data_cov = np.load(output_dir / "data_cov.npy")
    posterior_samples = np.load(output_dir / "posterior_samples.npy")

    return {
        "sigma_values": sigma_values,
        "pvalues": pvalues,
        "covariance": data_cov,
        "ground_truth": ground_truth,
        "data_points": data_values,
        "data_values": data_values,
        "data_cov": data_cov,
        "posterior_samples": posterior_samples,
    }


def plot_sweep_summary(output_dir: Path, sigma_values: np.ndarray, pvalues: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(sigma_values, pvalues, color="tab:blue", linewidth=2.5, label="PTED")
    # ax.scatter(sigma_values[[0, -1]], pvalues[[0, -1]], color="tab:blue", s=18)
    hdp_pvalue = np.load(output_dir / "pvalues_hdp.npy")
    ax.plot(sigma_values, hdp_pvalue, color="tab:green", linewidth=2.5, label="HPD")
    # ax.scatter(sigma_values[[0, -1]], hdp_pvalue[[0, -1]], color="tab:green", s=18)
    ks_pvalues = np.load(output_dir / "pvalues_ks.npy")
    ax.plot(sigma_values, ks_pvalues, color="tab:orange", linewidth=2.5, label="KS")
    mmd_pvalues = np.load(output_dir / "pvalues_mmd.npy")
    ax.plot(sigma_values, mmd_pvalues, color="tab:cyan", linewidth=2.5, label="MMD")
    mira_pvalue = np.load(output_dir / "pvalues_mira.npy")
    stdband = np.sqrt((1 / 18) / 64)
    mira_bounds = np.interp([2 / 3 - stdband, 2 / 3 + stdband], mira_pvalue, sigma_values)
    ax.axvline(mira_bounds[0], linestyle=":", color="tab:red", alpha=0.65, label="MIRA window")
    ax.axvline(mira_bounds[1], linestyle=":", color="tab:red", alpha=0.65)
    # ax_mira = ax.twinx()
    # (line_mira,) = ax_mira.plot(
    #     sigma_values,
    #     mira_pvalue,
    #     color="tab:red",
    #     linewidth=2.5,
    #     label="MIRA coverage score",
    # )
    # ax_mira.set_ylabel("MIRA score", color="tab:red")
    # ax_mira.tick_params(axis="y", colors="tab:red")
    # ax_mira.axhline(2 / 3, linestyle=":", color="tab:red", alpha=0.65, label="2/3 threshold")
    # ax_mira.axhline(
    #     2 / 3 - np.sqrt(varband), linestyle=":", color="tab:grey", alpha=0.65, label="MIRA window"
    # )
    # ax_mira.axhline(2 / 3 + np.sqrt(varband), linestyle=":", color="tab:grey", alpha=0.65)

    ax.axvline(1.0, linestyle="--", color="tab:gray", label="True sigma")
    # ax.axhline(0.05, linestyle=":", color="tab:red", alpha=0.65, label="0.05 threshold")

    ax.set_xscale("log")
    ax.set_xlabel("Posterior sigma scale [scale * $\\Sigma_{\\rm true}$]")
    ax.set_ylabel("coverage p-value")
    ax.set_title("PTED posterior coverage sensitivity sweep")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(sigma_values.min(), sigma_values.max())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(
        [sigma_values.min(), 0.8, 1.0, 1.2, sigma_values.max()],
        labels=[f"{sigma_values.min():.2f}", "0.8", "1.0", "1.2", f"{sigma_values.max():.2f}"],
    )
    # ax.grid(alpha=0.2)
    # handles1, labels1 = ax.get_legend_handles_labels()
    # handles2, labels2 = ax_mira.get_legend_handles_labels()
    # ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=10)
    ax.legend(loc="upper left", fontsize=10)
    path = output_dir / "coverage_sweep_summary.pdf"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_case_comparison(
    output_dir: Path,
    data_values: np.ndarray,
    posterior_samples: np.ndarray,
    ground_truth: np.ndarray,
    sigma_values: np.ndarray,
) -> Path:
    correct_sigma = float(1.0)
    under_sigma = float(sigma_values.max())
    over_sigma = float(sigma_values.min())

    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(9.0, 15.0))
    for row_idx in range(5):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            scaled_posterior_samples = (
                posterior_samples[:, row_idx + 1, :] - data_values[None, row_idx + 1, :]
            ) * ([over_sigma, correct_sigma, under_sigma][col_idx]) + data_values[
                None, row_idx + 1, :
            ]
            ax.scatter(
                scaled_posterior_samples[:, 0],
                scaled_posterior_samples[:, 1],
                s=10,
                color="tab:blue",
            )
            ax.scatter(
                ground_truth[row_idx + 1, 0],
                ground_truth[row_idx + 1, 1],
                s=55,
                color="tab:orange",
                marker="o",
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()
            if row_idx == 0:
                ax.set_title(
                    ["Overconfident", "Correct Posterior", "Underconfident"][col_idx], fontsize=18
                )
            if col_idx == 2:
                xcenter = (
                    scaled_posterior_samples[:, 0].min() + scaled_posterior_samples[:, 0].max()
                ) / 2
                ycenter = (
                    scaled_posterior_samples[:, 1].min() + scaled_posterior_samples[:, 1].max()
                ) / 2
                xwidth = scaled_posterior_samples[:, 0].max() - scaled_posterior_samples[:, 0].min()
                ywidth = scaled_posterior_samples[:, 1].max() - scaled_posterior_samples[:, 1].min()
                width = max(xwidth, ywidth)
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.set_xlim(xcenter - width / 2, xcenter + width / 2)
            ax.set_ylim(ycenter - width / 2, ycenter + width / 2)

    # fig.suptitle("Three posterior regimes: correct, underconfident, and overconfident")
    # fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = output_dir / "coverage_case_comparison.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def pit_plot(
    output_dir: Path,
    ground_truth: np.ndarray,
    posterior_samples: np.ndarray,
):
    torch.manual_seed(42)
    pted_coverage_test(
        torch.tensor(ground_truth),
        torch.tensor(posterior_samples),
        permutations=512,
        pit_plot=str(output_dir / "pit_plot.pdf"),
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _load_run_artifacts(output_dir)

    plot_sweep_summary(output_dir, artifacts["sigma_values"], artifacts["pvalues"])
    plot_case_comparison(
        output_dir=output_dir,
        data_values=artifacts["data_values"],
        posterior_samples=artifacts["posterior_samples"],
        ground_truth=artifacts["ground_truth"],
        sigma_values=artifacts["sigma_values"],
    )
    pit_plot(
        output_dir=output_dir,
        ground_truth=artifacts["ground_truth"],
        posterior_samples=artifacts["posterior_samples"],
    )
    print(f"Wrote summary plots to: {output_dir}")


if __name__ == "__main__":
    main()
