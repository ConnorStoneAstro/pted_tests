from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coverage_test.run_mira_null_test import summarize_null_scores


def plot_null_summary(output_dir: Path) -> Path:
    scores = np.load(output_dir / "scores_mira.npy")
    bootstrap_scores = np.load(output_dir / "bootstrap_scores_mira.npy")
    summary = summarize_null_scores(scores, bootstrap_scores)
    std = summary["bootstrap_std"]
    z_scores = (scores - 2 / 3) / std
    centered_bootstrap = bootstrap_scores - bootstrap_scores.mean() + 2 / 3
    x = np.linspace(
        min(scores.min(), centered_bootstrap.min(), 2 / 3 - 4 * std),
        max(scores.max(), centered_bootstrap.max(), 2 / 3 + 4 * std),
        500,
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    bins = np.histogram_bin_edges(np.concatenate((scores, centered_bootstrap)), bins="auto")
    ax.hist(scores, bins=bins, density=True, alpha=0.5, label="Independent null systems")
    ax.hist(
        centered_bootstrap, bins=bins, density=True, histtype="step",
        linewidth=1.8, label="Reference bootstrap (recentered to 2/3)",
    )
    ax.plot(x, norm.pdf(x, loc=2 / 3, scale=std), label="Normal(2/3, bootstrap std)")
    ax.axvline(2 / 3, color="gray", linestyle=":")
    ax.set(xlabel="MIRA score", ylabel="Density", title="Null distribution vs bootstrap prediction")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    quantiles = norm.ppf((np.arange(len(scores)) + 0.5) / len(scores))
    ax.scatter(quantiles, np.sort(z_scores), s=12)
    low = min(quantiles.min(), z_scores.min())
    high = max(quantiles.max(), z_scores.max())
    ax.plot([low, high], [low, high], color="gray", linestyle="--")
    ax.set(
        xlabel="Standard normal quantile",
        ylabel="Observed z = (MIRA - 2/3) / bootstrap std",
        title=f"Q-Q: z mean={summary['z_mean']:.2f}, std={summary['z_std']:.2f}",
    )

    ax = axes[1, 0]
    for values, label in (
        (scores, "Independent null systems"),
        (centered_bootstrap, "Reference bootstrap (recentered)"),
    ):
        values = np.sort(values)
        ax.step(
            np.r_[x[0], values, x[-1]],
            np.r_[0, np.arange(1, len(values) + 1) / len(values), 1],
            where="post", label=label,
        )
    ax.plot(x, norm.cdf(x, loc=2 / 3, scale=std), label="Normal prediction")
    ax.set(xlabel="MIRA score", ylabel="Cumulative probability", title="Distribution shape comparison")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    multipliers = np.arange(1, 4)
    observed = np.array([item["observed_fraction"] for item in summary["coverage"]])
    expected = [item["gaussian_expected_fraction"] for item in summary["coverage"]]
    ax.bar(multipliers - 0.18, observed, width=0.36, label="Observed null coverage")
    ax.bar(multipliers + 0.18, expected, width=0.36, label="Gaussian prediction")
    ax.set_xticks(multipliers, labels=["+/- 1 std", "+/- 2 std", "+/- 3 std"])
    ax.set(
        ylim=(0, 1.12), ylabel="Fraction within band centered on 2/3",
        title=f"Empirical / bootstrap std = {summary['null_to_bootstrap_std_ratio']:.2f}",
    )
    ax.legend(fontsize=8)
    fig.suptitle(
        f"MIRA null calibration: {len(scores)} independent systems, "
        f"{len(bootstrap_scores)} reference bootstraps"
    )
    fig.tight_layout()
    path = output_dir / "mira_null_summary.pdf"
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MIRA null scores against reference bootstrap uncertainty")
    parser.add_argument("--output-dir", default="coverage_test/results/mira_null_test")
    args = parser.parse_args()
    print(f"Wrote MIRA null summary to: {plot_null_summary(Path(args.output_dir))}")


if __name__ == "__main__":
    main()
