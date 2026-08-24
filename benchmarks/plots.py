from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from benchmarks.metrics import metric_sweep

METHOD_COLOURS = {
    "pted": "tab:blue",
    "ks_pc1": "tab:orange",
    "fld": "tab:green",
    "fid": "tab:red",
    "pqm": "tab:purple",
}
METHOD_LABELS = {
    "pted": "PTED",
    "ks_pc1": "KS-PC1",
    "fld": "FLD",
    "fid": "FID",
    "pqm": "PQM",
}


def _progressive_linewidth(
    index: int, total: int, max_width: float = 2.6, min_width: float = 1.1
) -> float:
    if total <= 1:
        return max_width
    t = index / (total - 1)
    return max_width + t * (min_width - max_width)


def _group_records(
    records: Iterable[dict[str, object]],
    x_key: str = "severity",
    y_key: str = "score",
) -> dict[str, dict[float, list[float]]]:
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        method = str(record["method"])
        x_value = float(record[x_key])
        if y_key not in record:
            continue
        score = float(record[y_key])
        if np.isfinite(score):
            grouped[method][x_value].append(score)
    for method in grouped:
        grouped[method] = (
            np.sort(list(grouped[method].keys())),
            np.stack([grouped[method][x] for x in np.sort(list(grouped[method].keys()))], axis=0),
        )
    return grouped  # dict[str, tuple[np.ndarray, np.ndarray]]


def plot_method_sweep(
    records: Iterable[dict[str, object]],
    output_path: str | Path,
    title: str,
    x_key: str = "severity",
    y_key: str = "score",
    ylabel: str = "score",
    new_yscale: Sequence[str] = [],
    decision_threshold: float | None = None,
    lower_quantile: float = 0.16,
    upper_quantile: float = 0.84,
    max_linewidth: float = 2.6,
    min_linewidth: float = 1.1,
):
    for metric in metric_sweep().keys():
        subset = [record for record in records if record["method"] == metric]
        print(
            f"Metric: {metric} avg runtime: {sum(record['runtime'] for record in subset)/len(subset):.4e} seconds"
        )

    grouped = _group_records(records, x_key=x_key, y_key=y_key)
    methods = list(reversed(sorted(grouped)))

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    total_methods = len(methods)
    count_extra = 0
    lines = []
    for i, method in enumerate(methods):
        x_values = grouped[method][0]
        medians = np.median(grouped[method][1], axis=1)
        lowers = np.quantile(grouped[method][1], lower_quantile, axis=1)
        uppers = np.quantile(grouped[method][1], upper_quantile, axis=1)
        linewidth = _progressive_linewidth(
            i, total_methods, max_width=max_linewidth, min_width=min_linewidth
        )
        if method in new_yscale:
            count_extra += 1
            subax = ax.twinx()
            subax.tick_params(axis="y", labelcolor=METHOD_COLOURS[method])
            if count_extra > 1:
                subax.spines["right"].set_position(("axes", 1 + 0.1 * (count_extra - 1)))
            useax = subax
        else:
            useax = ax
        useax.fill_between(x_values, lowers, uppers, alpha=0.15, color=METHOD_COLOURS[method])
        (line,) = useax.plot(
            x_values,
            medians,
            label=METHOD_LABELS[method],
            linewidth=linewidth,
            color=METHOD_COLOURS[method],
        )
        lines.append(line)

    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlim(np.min(x_values), np.max(x_values))
    ax.set_xlabel("Severity score [S]")
    ax.set_ylabel(ylabel)
    if decision_threshold is not None:
        ax.axhline(decision_threshold, color="k", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.legend(handles=lines, loc="upper center")
    ax.grid(True, alpha=0.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
