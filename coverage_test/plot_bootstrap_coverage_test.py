from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_sweep_summary(output_dir: Path) -> Path:
    sigma_values = np.load(output_dir / "sigma_values.npy")
    pvalues_pted = np.load(output_dir / "pvalues_pted.npy")
    scores_mira = np.load(output_dir / "scores_mira.npy")
    bootstrap_std = np.load(output_dir / "bootstrap_std_mira.npy")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax_mira = ax.twinx()
    ax.plot(sigma_values, pvalues_pted, color="tab:blue", linewidth=2.5, label="PTED")
    ax_mira.plot(
        sigma_values, scores_mira, color="tab:red", linewidth=2.5, label="MIRA score"
    )
    ax_mira.axhline(
        2 / 3, linestyle=":", color="tab:red", label="MIRA 2/3 threshold"
    )
    ax_mira.fill_between(
        sigma_values,
        2 / 3 - bootstrap_std,
        2 / 3 + bootstrap_std,
        color="tab:red",
        alpha=0.2,
        label="MIRA 2/3 +/- 1 bootstrap std",
    )
    ax.axvline(1.0, linestyle="--", color="tab:gray", label="True sigma")
    ax.set_xlabel("Posterior sigma scale")
    ax.set_ylabel("PTED coverage p-value", color="tab:blue")
    ax.tick_params(axis="y", colors="tab:blue")
    ax.set_ylim(0.0, 1.0)
    ax_mira.set_ylabel("MIRA score", color="tab:red")
    ax_mira.tick_params(axis="y", colors="tab:red")
    ax.set_xticks(sigma_values)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_title("PTED and bootstrap-calibrated MIRA coverage sweep")
    handles, labels = ax.get_legend_handles_labels()
    mira_handles, mira_labels = ax_mira.get_legend_handles_labels()
    ax.legend(handles + mira_handles, labels + mira_labels, loc="best", fontsize=9)
    fig.tight_layout()
    path = output_dir / "bootstrap_coverage_sweep_summary.pdf"
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PTED and bootstrap-calibrated MIRA results")
    parser.add_argument("--output-dir", default="coverage_test/results/bootstrap_coverage_test")
    args = parser.parse_args()
    print(f"Wrote bootstrap coverage summary to: {plot_sweep_summary(Path(args.output_dir))}")


if __name__ == "__main__":
    main()
