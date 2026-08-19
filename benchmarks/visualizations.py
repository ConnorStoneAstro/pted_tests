"""Explanatory figures showing what each benchmark deviation does to the data.

Reads the same suite configuration files used by the runners so the severity
grids in the figures match the ones actually benchmarked.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.datasets.vision import load_vision_dataset
from benchmarks.utils import _grab_config

SEVERITY_CMAP = "viridis"
BASELINE_COLOUR = "black"

GAUSSIAN_TITLES = {
    "mean_shift": "$y \\sim \\mathcal{N}({\\rm S}, 1)$",
    "scale_shift": "$y \\sim \\mathcal{N}(0, (1 + {\\rm S})^2)$",
    "bimodal": "$y \\sim 0.5 \\mathcal{N}(-{\\rm S}, 1) + 0.5 \\mathcal{N}({\\rm S}, 1)$",
    "contamination": "$y \\sim (1 - {\\rm S}) \\mathcal{N}(0, 1) + {\\rm S} \\mathcal{N}(4, 1)$",
}
VISION_TITLES = {
    "pair_blend": "$y = (1 - \\frac{{\\rm S}}{2}) y_1 + \\frac{{\\rm S}}{2} y_2$",
    "class_drop": "class 0 reweighted by $(1 - {\\rm S})$",
    "white_noise": "$y = y_1 + \\mathcal{N}(0, (\\frac{{\\rm S}}{3})^2)$",
}
DATASET_NAMES = {
    "mnist": "MNIST",
    "cifar10": "CIFAR-10",
    "gaussian2x2": "Gaussian 2x2",
}


def _normal_pdf(grid: np.ndarray, loc: float, scale: float) -> np.ndarray:
    return np.exp(-0.5 * ((grid - loc) / scale) ** 2) / (scale * np.sqrt(2.0 * np.pi))


def _gaussian_density(grid: np.ndarray, deviation: str, severity: float) -> np.ndarray:
    if deviation == "mean_shift":
        return _normal_pdf(grid, severity, 1.0)
    if deviation == "scale_shift":
        return _normal_pdf(grid, 0.0, 1.0 + severity)
    if deviation == "bimodal":
        return 0.5 * _normal_pdf(grid, -severity, 1.0) + 0.5 * _normal_pdf(grid, severity, 1.0)
    if deviation == "contamination":
        return (1.0 - severity) * _normal_pdf(grid, 0.0, 1.0) + severity * _normal_pdf(
            grid, 4.0, 1.0
        )
    raise ValueError(f"Unknown deviation kind: {deviation}")


def _severity_colours(severities: np.ndarray) -> tuple[ScalarMappable, np.ndarray]:
    norm = Normalize(vmin=float(severities.min()), vmax=float(severities.max()))
    mappable = ScalarMappable(norm=norm, cmap=plt.get_cmap(SEVERITY_CMAP))
    mappable.set_array(severities)
    return mappable, mappable.to_rgba(severities)


def _to_displayable(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return image


def plot_gaussian_deviation(
    deviation: str,
    severities: np.ndarray,
    output_path: Path,
) -> None:
    if deviation == "contamination":
        grid = np.linspace(-4.0, 8.0, 1024)
    else:
        grid = np.linspace(-6.0, 6.0, 1024)

    mappable, colours = _severity_colours(severities)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        grid,
        _normal_pdf(grid, 0.0, 1.0),
        color=BASELINE_COLOUR,
        linewidth=2.2,
        linestyle="--",
        label="$x \\sim \\mathcal{N}(0, 1)$",
        zorder=3,
    )
    for severity, colour in zip(severities, colours):
        ax.plot(
            grid,
            _gaussian_density(grid, deviation, float(severity)),
            color=colour,
            linewidth=1.6,
        )

    ax.set_xlabel("$x$")
    ax.set_ylabel("density")
    ax.set_title(f"Gaussian 1D: {GAUSSIAN_TITLES[deviation]}")
    ax.legend(loc="upper right", frameon=False)
    fig.colorbar(mappable, ax=ax, label="severity ${\\rm S}$")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _image_grid(
    panels: np.ndarray,
    severities: np.ndarray,
    title: str,
    output_path: Path,
    row_labels: list[str] | None = None,
) -> None:
    """panels has shape (n_rows, n_severities, ...) of displayable images."""
    n_rows, n_cols = panels.shape[0], panels.shape[1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.15 * n_cols + 1.6, 1.15 * n_rows + 1.0),
        squeeze=False,
    )
    mappable, colours = _severity_colours(severities)
    is_rgb = panels.ndim == 5 and panels.shape[-1] == 3
    if is_rgb:
        panels = np.clip(panels, 0.0, 1.0)
    vmin = float(np.min(panels[:, 0]))
    vmax = float(np.max(panels[:, 0]))

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row][col]
            if is_rgb:
                ax.imshow(panels[row, col], interpolation="nearest")
            else:
                ax.imshow(
                    panels[row, col], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest"
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(colours[col])
                spine.set_linewidth(2.0)
            if row == 0:
                ax.set_title(f"S = {severities[col]:.2f}", fontsize=8, color=colours[col])
        if row_labels is not None:
            axes[row][0].set_ylabel(row_labels[row], fontsize=8)

    fig.suptitle(title)
    fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        label="severity ${\\rm S}$",
        fraction=0.03,
        pad=0.02,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_vision_white_noise(
    images: np.ndarray,
    severities: np.ndarray,
    dataset: str,
    output_path: Path,
    n_examples: int = 3,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(images), size=n_examples, replace=False)
    panels = []
    for index in picks:
        base = _to_displayable(images[index])
        noise = rng.normal(size=base.shape)
        panels.append([base + noise * (float(s) / 3.0) for s in severities])
    _image_grid(
        np.asarray(panels),
        severities,
        title=f"{DATASET_NAMES.get(dataset, dataset)} white noise: {VISION_TITLES['white_noise']}",
        output_path=output_path,
    )


def plot_vision_pair_blend(
    images: np.ndarray,
    severities: np.ndarray,
    dataset: str,
    output_path: Path,
    n_examples: int = 3,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(images), size=2 * n_examples, replace=False)
    panels = []
    for first, second in zip(picks[:n_examples], picks[n_examples:]):
        y1 = _to_displayable(images[first])
        y2 = _to_displayable(images[second])
        panels.append([(1.0 - float(s) / 2.0) * y1 + (float(s) / 2.0) * y2 for s in severities])
    _image_grid(
        np.asarray(panels),
        severities,
        title=f"{DATASET_NAMES.get(dataset, dataset)} pair blend: {VISION_TITLES['pair_blend']}",
        output_path=output_path,
    )


def plot_vision_class_drop(
    labels: np.ndarray,
    severities: np.ndarray,
    dataset: str,
    output_path: Path,
) -> None:
    classes, counts = np.unique(labels, return_counts=True)
    mappable, colours = _severity_colours(severities)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    width = 0.8 / len(severities)
    positions = np.arange(len(classes))
    for index, (severity, colour) in enumerate(zip(severities, colours)):
        weights = np.ones(len(classes))
        weights[0] = 1.0 - float(severity)
        proportions = weights * counts
        proportions = proportions / proportions.sum()
        ax.bar(
            positions - 0.4 + (index + 0.5) * width,
            proportions,
            width=width,
            color=colour,
        )
    ax.axhline(
        1.0 / len(classes) if np.allclose(counts, counts[0]) else float((counts / counts.sum())[0]),
        color=BASELINE_COLOUR,
        linestyle="--",
        linewidth=1.2,
        label="baseline class 0 fraction",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_xlabel("class")
    ax.set_ylabel("sampling proportion")
    ax.set_title(f"{DATASET_NAMES.get(dataset, dataset)} class drop: {VISION_TITLES['class_drop']}")
    ax.legend(loc="upper right", frameon=False)
    fig.colorbar(mappable, ax=ax, label="severity ${\\rm S}$")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def visualize_gaussian_suite(config: dict, out_dir: Path) -> None:
    severities = np.asarray(config["severities"], dtype=float)
    for deviation in config["deviations"]:
        path = out_dir / f"gaussian_1d_{deviation}.png"
        plot_gaussian_deviation(deviation, severities, path)
        print(f"Wrote figure: {path}")


def visualize_vision_suite(config: dict, out_dir: Path) -> None:
    severities = np.asarray(config["severities"], dtype=float)
    for dataset in config["datasets"]:
        images, labels = load_vision_dataset(
            dataset,
            data_root=config.get("data_root"),
            download=bool(config.get("download", False)),
        )
        for deviation in config["deviations"]:
            path = out_dir / f"vision_{dataset}_{deviation}.png"
            if deviation == "white_noise":
                plot_vision_white_noise(images, severities, dataset, path)
            elif deviation == "pair_blend":
                plot_vision_pair_blend(images, severities, dataset, path)
            elif deviation == "class_drop":
                plot_vision_class_drop(labels, severities, dataset, path)
            else:
                raise ValueError(f"Unknown deviation kind: {deviation}")
            print(f"Wrote figure: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate explanatory figures for the benchmark deviations"
    )
    parser.add_argument(
        "--gaussian-config",
        default="benchmarks/configs/gaussian_suite.py",
        help="Gaussian suite configuration file (set to 'none' to skip)",
    )
    parser.add_argument(
        "--vision-config",
        default="benchmarks/configs/vision_suite.py",
        help="Vision suite configuration file (set to 'none' to skip)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results/visualizations",
        help="Directory to write the figures into",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir)

    if args.gaussian_config.lower() != "none":
        visualize_gaussian_suite(_grab_config(args.gaussian_config), out_dir)
    if args.vision_config.lower() != "none":
        visualize_vision_suite(_grab_config(args.vision_config), out_dir)


if __name__ == "__main__":
    main()
