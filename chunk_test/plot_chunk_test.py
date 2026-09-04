from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _group_by_chunk(
    records: Iterable[dict[str, object]],
    x_key: str = "severity",
    y_key: str = "score",
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    buckets: dict[int, dict[float, list[float]]] = {}
    for record in records:
        chunk = int(float(record["chunk_size"]))
        x_value = float(record[x_key])
        buckets.setdefault(chunk, {}).setdefault(x_value, []).append(float(record[y_key]))

    grouped: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for chunk, by_x in buckets.items():
        x_values = np.sort(np.array(list(by_x.keys()), dtype=float))
        scores = np.stack([by_x[x] for x in x_values], axis=0)
        grouped[chunk] = (x_values, scores)
    return grouped


def plot_chunk_sweep(
    records: Sequence[dict[str, object]],
    output_path: str | Path,
    title: str,
    x_key: str = "severity",
    y_key: str = "score",
    ylabel: str = "p-value",
    lower_quantile: float = 0.16,
    upper_quantile: float = 0.84,
    decision_threshold: float | None = 0.05,
) -> None:
    grouped = _group_by_chunk(records, x_key=x_key, y_key=y_key)
    chunk_sizes = sorted(grouped)

    cmap = plt.get_cmap("Blues")
    # Keep the light end readable against white.
    colours = [cmap(0.35 + 0.6 * i / max(len(chunk_sizes) - 1, 1)) for i in range(len(chunk_sizes))]

    fig, ax = plt.subplots(figsize=(8, 5))
    for colour, chunk in zip(colours, chunk_sizes):
        x_values, scores = grouped[chunk]
        medians = np.nanmedian(scores, axis=1)
        lowers = np.nanquantile(scores, lower_quantile, axis=1)
        uppers = np.nanquantile(scores, upper_quantile, axis=1)
        ax.fill_between(x_values, lowers, uppers, alpha=0.12, color=colour, linewidth=0)
        ax.plot(x_values, medians, color=colour, linewidth=2.0, label=f"chunk = {chunk}")

    if decision_threshold is not None:
        ax.axhline(decision_threshold, color="k", linestyle="--", linewidth=1.5, alpha=0.7)

    all_x = np.concatenate([grouped[chunk][0] for chunk in chunk_sizes])
    ax.set_title(title)
    ax.set_xlim(float(np.min(all_x)), float(np.max(all_x)))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Severity score [S]")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper center", ncol=2)
    ax.grid(True, alpha=0.2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_chunk_runtime(
    records: Sequence[dict[str, object]],
    output_path: str | Path,
    title: str,
) -> None:
    buckets: dict[int, list[float]] = {}
    for record in records:
        buckets.setdefault(int(float(record["chunk_size"])), []).append(float(record["runtime"]))
    chunk_sizes = sorted(buckets)

    medians = [float(np.median(buckets[chunk])) for chunk in chunk_sizes]
    lowers = [float(np.quantile(buckets[chunk], 0.16)) for chunk in chunk_sizes]
    uppers = [float(np.quantile(buckets[chunk], 0.84)) for chunk in chunk_sizes]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(chunk_sizes, lowers, uppers, alpha=0.15, color="tab:blue", linewidth=0)
    ax.plot(chunk_sizes, medians, color="tab:blue", marker="o", linewidth=2.0)
    ax.set_xscale("log", base=2)
    # ax.set_yscale("log")
    ax.set_xlabel("chunk size")
    ax.set_ylabel("runtime [s]")
    ax.set_title(title)
    ax.grid(True, alpha=0.2, which="both")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
