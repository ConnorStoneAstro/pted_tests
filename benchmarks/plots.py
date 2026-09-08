from __future__ import annotations

from collections import defaultdict
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
    "mmd": "tab:cyan",
}
METHOD_LABELS = {
    "pted": "PTED",
    "ks_pc1": "KS-PC1",
    "fld": "FLD",
    "fid": "FID",
    "pqm": "PQM",
    "mmd": "MMD",
}

SENSITIVITY_PVALUE = 0.05


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
    max_linewidth: float = 3,
    min_linewidth: float = 1.5,
):
    for metric in metric_sweep().keys():
        subset = [record for record in records if record["method"] == metric]
        print(
            f"Metric: {metric} avg runtime: {sum(float(record.get('runtime', 0)) for record in subset)/len(subset):.4e} seconds"
        )

    grouped = _group_records(records, x_key=x_key, y_key=y_key)
    methods = list(reversed(sorted(grouped)))

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    total_methods = len(methods)
    count_extra = 0
    lines = []
    for i, method in enumerate(methods):
        print(method)
        print(np.sum(np.isfinite(grouped[method][1]), axis=1))
        x_values = grouped[method][0]
        medians = np.nanmedian(grouped[method][1], axis=1)
        lowers = np.nanquantile(grouped[method][1], lower_quantile, axis=1)
        uppers = np.nanquantile(grouped[method][1], upper_quantile, axis=1)
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
        useax.fill_between(
            x_values, lowers, uppers, alpha=0.05, color=METHOD_COLOURS[method], linewidth=0
        )
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


def _interpolate_crossing(
    severities: np.ndarray,
    values: np.ndarray,
    limit: float,
    crosses_upward: bool,
) -> float:
    """Return the first linearly interpolated threshold crossing, or 1.0."""
    finite = np.isfinite(values)
    severities = severities[finite]
    values = values[finite]
    if len(severities) == 0:
        return 1.0

    crosses_limit = values >= limit if crosses_upward else values <= limit
    if crosses_limit[0]:
        return float(severities[0])

    for index in range(1, len(severities)):
        if not crosses_limit[index]:
            continue
        previous_severity, severity = severities[index - 1 : index + 1]
        previous_value, value = values[index - 1 : index + 1]
        if np.isclose(value, previous_value):
            return float(severity)
        fraction = (limit - previous_value) / (value - previous_value)
        return float(previous_severity + fraction * (severity - previous_severity))
    return 1.0


def _condition_label(records: Sequence[dict[str, object]]) -> str:
    first = records[0]
    deviation = str(first["deviation"]).replace("_", " ")
    if "dataset" not in first or not first["dataset"]:
        return f"Gaussian 1D\n{deviation}"
    dataset = str(first["dataset"])
    dataset_names = {"gaussian2x2": "Gaussian 2x2", "mnist": "MNIST", "cifar10": "CIFAR-10"}
    return f"{dataset_names.get(dataset, dataset)}\n{deviation}"


def plot_sensitivity_thresholds(
    records: Iterable[dict[str, object]], output_path: str | Path
) -> None:
    """Plot each metric's severity required to detect every benchmark deviation."""
    condition_records: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        dataset = str(record.get("dataset", "gaussian_1d"))
        condition_records[(dataset, str(record["deviation"]))].append(record)

    if not condition_records:
        raise ValueError("No benchmark records were supplied for the sensitivity summary")

    conditions = sorted(condition_records)
    labels = [_condition_label(condition_records[condition]) for condition in conditions]
    methods = [
        method
        for method in METHOD_LABELS
        if any(method in _group_records(condition_records[condition]) for condition in conditions)
    ]
    thresholds: dict[str, list[float]] = {method: [] for method in methods}
    for condition in conditions:
        by_method = _group_records(condition_records[condition])
        for method in methods:
            if method not in by_method:
                thresholds[method].append(1.0)
                continue
            severities, scores = by_method[method]
            if not np.any(np.isfinite(scores)):
                thresholds[method].append(1.0)
                continue
            medians = np.nanmedian(scores, axis=1)
            if method in {"fld", "fid"}:
                null_scores = scores[np.isclose(severities, 0.0)]
                if null_scores.size == 0:
                    threshold = 1.0
                else:
                    upper_null_limit = float(np.nanquantile(null_scores, 0.975))
                    threshold = _interpolate_crossing(
                        severities, medians, upper_null_limit, crosses_upward=True
                    )
            else:
                threshold = _interpolate_crossing(
                    severities, medians, SENSITIVITY_PVALUE, crosses_upward=False
                )
            thresholds[method].append(threshold)

    positions = np.arange(len(conditions))
    N = np.argsort(thresholds["pted"])
    fig, ax = plt.subplots(figsize=(max(8.0, 0.8 * len(conditions)), 4.0))
    for i, method in enumerate(methods):
        marker_size = 200 - 30 * i
        ax.scatter(
            positions,
            np.array(thresholds[method])[N],
            s=marker_size,
            color=METHOD_COLOURS[method],
            label=METHOD_LABELS[method],
            edgecolors="white",
            linewidths=0.0,
            # zorder=3 if method == "pted" else 2,
        )
        if method == "pted":
            ax.plot(
                positions,
                np.array(thresholds[method])[N],
                color=METHOD_COLOURS[method],
                linewidth=2.0,
                alpha=0.5,
                zorder=0,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(np.array(labels)[N], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.2, len(conditions) - 0.8)
    ax.set_ylim(0.0, 1.03)
    # ax.set_xlabel("Benchmark test")
    ax.set_ylabel("Severity threshold [S, lower is better]")
    ax.set_title(r"Two-sample test comparison benchmark (95% sensitivity)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
