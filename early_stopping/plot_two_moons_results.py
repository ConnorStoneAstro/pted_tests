from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from early_stopping.two_moons import two_moons_density


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    pted_pvalue: float
    effective_train_subset_size: int
    best_epoch_so_far: int
    best_pvalue_so_far: float
    suggested_early_stop_epoch: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render plots and GIF from saved two-moons early-stopping artifacts"
    )
    parser.add_argument("--output-dir", default="early_stopping/results/two_moons")
    parser.add_argument("--plot-summary", action="store_true")
    parser.add_argument("--plot-epochs", action="store_true")
    parser.add_argument("--make-gif", action="store_true")
    parser.add_argument("--gif-name", default="epoch_animation.gif")
    parser.add_argument("--gif-fps", type=float, default=6.0)
    parser.add_argument("--gif-loop", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    return parser


def _read_config(output_dir: Path) -> dict[str, object]:
    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_history(output_dir: Path) -> list[EpochRecord]:
    history_path = output_dir / "history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history CSV: {history_path}")

    rows: list[EpochRecord] = []
    with history_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                EpochRecord(
                    epoch=int(row["epoch"]),
                    train_loss=float(row["train_loss"]),
                    pted_pvalue=float(row["pted_pvalue"]),
                    effective_train_subset_size=int(
                        row.get("effective_train_subset_size", row.get("train_set_size", 0))
                    ),
                    best_epoch_so_far=int(row["best_epoch_so_far"]),
                    best_pvalue_so_far=float(row["best_pvalue_so_far"]),
                    suggested_early_stop_epoch=int(row["suggested_early_stop_epoch"]),
                )
            )

    if not rows:
        raise ValueError(f"History CSV is empty: {history_path}")
    return rows


def _load_density_grid(output_dir: Path, noise: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = output_dir / "reference_density.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["xx"], cached["yy"], cached["density"]

    xx, yy = np.meshgrid(
        np.linspace(-1.8, 2.8, 220),
        np.linspace(-1.4, 1.8, 220),
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    density = two_moons_density(grid, noise=noise, quadrature_points=512).reshape(xx.shape)
    return xx.astype(np.float32), yy.astype(np.float32), density.astype(np.float32)


def _plot_summary(output_dir: Path, history: list[EpochRecord]) -> Path:
    epochs = np.array([r.epoch for r in history], dtype=int)
    pvalues = np.array([r.pted_pvalue for r in history], dtype=float)
    losses = np.array([r.train_loss for r in history], dtype=float)
    suggested = int(history[-1].suggested_early_stop_epoch)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)

    ax1.plot(epochs, pvalues, color="tab:blue", linewidth=2.0, marker="o", markersize=2.8)
    # ax1.axvline(suggested, color="tab:green", linestyle=":", linewidth=2.0)
    ax1.set_ylabel("PTED p-value")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(alpha=0.2)
    ax1.set_title("PTED Early-Stopping Signal Across Training")

    ax2.plot(epochs, losses, color="tab:red", linewidth=1.8, marker=".", markersize=3.0)
    # ax2.axvline(suggested, color="tab:green", linestyle=":", linewidth=2.0)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Train loss")
    ax2.grid(alpha=0.2)

    summary_path = output_dir / "pted_summary.png"
    fig.savefig(summary_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary_path


def _plot_epoch(
    output_path: Path,
    xx: np.ndarray,
    yy: np.ndarray,
    true_density: np.ndarray,
    train_data: np.ndarray,
    generated: np.ndarray,
    epoch: int,
    pvalue: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    contour = ax.contourf(xx, yy, true_density, levels=30, cmap="viridis")
    fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04, label="True density")

    ax.scatter(
        train_data[:, 0],
        train_data[:, 1],
        s=22,
        c="white",
        alpha=0.45,
        edgecolors="none",
        label="Train data",
    )
    ax.scatter(
        generated[:, 0],
        generated[:, 1],
        s=12,
        c="tomato",
        alpha=0.55,
        edgecolors="none",
        label="Generated samples",
    )

    ax.set_title(f"Two Moons Epoch {epoch:03d} | PTED p-value={pvalue:.4f}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right")
    ax.set_xlim(float(xx.min()), float(xx.max()))
    ax.set_ylim(float(yy.min()), float(yy.max()))
    ax.grid(alpha=0.15)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _collect_epoch_sample_files(samples_dir: Path) -> list[tuple[int, Path]]:
    pattern = re.compile(r"^epoch_(\d+)\.npy$")
    files: list[tuple[int, Path]] = []
    for path in samples_dir.glob("epoch_*.npy"):
        match = pattern.match(path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    if not files:
        raise FileNotFoundError(f"No epoch samples found in: {samples_dir}")
    return files


def _plot_epochs(output_dir: Path, history: list[EpochRecord], noise: float) -> list[Path]:
    plots_dir = output_dir / "epoch_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_data_path = output_dir / "train_data.npy"
    if not train_data_path.exists():
        raise FileNotFoundError(f"Missing training data artifact: {train_data_path}")
    train_data = np.load(train_data_path)

    xx, yy, true_density = _load_density_grid(output_dir=output_dir, noise=noise)

    history_map = {record.epoch: record for record in history}
    sample_files = _collect_epoch_sample_files(output_dir / "epoch_samples")

    output_paths: list[Path] = []
    for epoch, sample_file in sample_files:
        if epoch not in history_map:
            continue
        generated = np.load(sample_file)
        record = history_map[epoch]
        output_path = plots_dir / f"epoch_{epoch:03d}.png"
        _plot_epoch(
            output_path=output_path,
            xx=xx,
            yy=yy,
            true_density=true_density,
            train_data=train_data,
            generated=generated,
            epoch=epoch,
            pvalue=record.pted_pvalue,
        )
        output_paths.append(output_path)
    return output_paths


def _make_gif(
    png_paths: list[Path],
    gif_path: Path,
    fps: float,
    loop: int,
    frame_step: int,
) -> Path:
    if imageio is None:
        raise ImportError(
            "imageio is required to create GIF files. Install with: pip install imageio"
        )
    if frame_step <= 0:
        raise ValueError("frame_step must be >= 1")

    # Sort by numeric epoch extracted from the filename so frames stay in the
    # correct temporal order when epoch numbers exceed the filename zero-padding.
    pattern = re.compile(r"^epoch_(\d+)\.png$")

    def _png_epoch_key(path: Path) -> int:
        match = pattern.match(path.name)
        if match:
            return int(match.group(1))
        return 10**12

    ordered_png_paths = sorted(png_paths, key=_png_epoch_key)

    selected_paths = ordered_png_paths[::frame_step]
    if not selected_paths:
        raise ValueError("No PNG frames selected for GIF creation")

    frames = [imageio.imread(path) for path in selected_paths]
    frame_duration = 1.0 / max(fps, 1e-6)
    imageio.mimsave(gif_path, frames, duration=frame_duration, loop=loop)
    return gif_path


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    if not args.plot_summary and not args.plot_epochs and not args.make_gif:
        args.plot_summary = True
        args.plot_epochs = True
        args.make_gif = True

    config = _read_config(output_dir)
    history = _read_history(output_dir)
    noise = float(config.get("noise", 0.08))

    if args.plot_summary:
        summary_path = _plot_summary(output_dir=output_dir, history=history)
        print(f"Wrote summary plot: {summary_path}")

    epoch_png_paths: list[Path] = []
    if args.plot_epochs or args.make_gif:
        epoch_png_paths = _plot_epochs(output_dir=output_dir, history=history, noise=noise)
        print(f"Wrote {len(epoch_png_paths)} epoch plots to: {output_dir / 'epoch_plots'}")

    if args.make_gif:
        gif_path = _make_gif(
            png_paths=epoch_png_paths,
            gif_path=output_dir / args.gif_name,
            fps=args.gif_fps,
            loop=args.gif_loop,
            frame_step=args.frame_step,
        )
        print(f"Wrote GIF: {gif_path}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
