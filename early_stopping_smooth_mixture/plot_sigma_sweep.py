from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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

matplotlib.rcParams.update({"font.size": 14, "axes.labelsize": 14, "axes.titlesize": 14})

P_VALUE_METHODS = {"pted", "ks_pc1", "pqm", "mmd"}
METHOD_COLORS = {
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot smooth two-moons sigma sweep frames with side-by-side sample and metric panels"
        )
    )
    parser.add_argument(
        "--output-dir",
        default="early_stopping_smooth_mixture/results/two_moons_sigma_sweep",
    )
    parser.add_argument("--plot-frames", action="store_true")
    parser.add_argument("--make-gif", action="store_true")
    parser.add_argument("--gif-name", default="sigma_sweep_animation.gif")
    parser.add_argument("--gif-fps", type=float, default=8.0)
    parser.add_argument("--gif-loop", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    return parser


def _load_metrics(metrics_path: Path) -> dict[str, list[tuple[int, float, float]]]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing sweep metrics CSV: {metrics_path}")

    grouped: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sigma_idx = int(row["sigma_idx"])
            sigma = float(row["sigma"])
            method = str(row["method"])
            score = float(row["score"])
            grouped[method].append((sigma_idx, sigma, score))

    for method, rows in grouped.items():
        rows.sort(key=lambda item: item[0])

    if not grouped:
        raise ValueError(f"No metric records found in {metrics_path}")

    return grouped


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


def _plot_frame(
    frame_path: Path,
    frame_idx: int,
    sigma: float,
    train_data: np.ndarray,
    generated: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    density: np.ndarray,
    metrics_by_method: dict[str, list[tuple[int, float, float]]],
    sigma_values: np.ndarray,
    pvalue_ylim: tuple[float, float],
    fld_ylim: tuple[float, float],
    fid_ylim: tuple[float, float],
    true_noise: float | None = None,
) -> None:
    fig = plt.figure(figsize=(12.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.2)

    ax_samples = fig.add_subplot(gs[0, 0])
    contour = ax_samples.contourf(xx, yy, density, levels=30, cmap="viridis")
    ax_samples.scatter(
        train_data[:, 0],
        train_data[:, 1],
        s=18,
        c="white",
        alpha=0.35,
        edgecolors="none",
        label="Train data",
    )
    ax_samples.scatter(
        generated[:, 0],
        generated[:, 1],
        s=9,
        c="tomato",
        alpha=0.50,
        edgecolors="none",
        label="Mixture samples",
    )
    ax_samples.set_title(f"Sigma = {sigma:.3f}")
    ax_samples.set_xlim(float(xx.min()), float(xx.max()))
    ax_samples.set_ylim(float(yy.min()), float(yy.max()))
    ax_samples.set_xlabel("")
    ax_samples.set_ylabel("")
    ax_samples.set_xticks([])
    ax_samples.set_yticks([])
    ax_samples.legend(loc="upper right")

    ax_p = fig.add_subplot(gs[0, 1])
    extra_axes: dict[str, plt.Axes] = {}
    count_extra = 0
    for method in ("fld", "fid"):
        if method not in metrics_by_method:
            continue
        count_extra += 1
        subax = ax_p.twinx()
        subax.tick_params(axis="y", labelcolor=METHOD_COLORS[method])
        if count_extra > 1:
            subax.spines["right"].set_position(("axes", 1 + 0.15 * (count_extra - 1)))
        subax.set_ylabel(f"{METHOD_LABELS[method]} score", color=METHOD_COLORS[method])
        extra_axes[method] = subax

    lines = []
    labels = []
    for method, rows in sorted(metrics_by_method.items()):
        x_vals = np.array([item[1] for item in rows], dtype=float)
        y_vals = np.array([item[2] for item in rows], dtype=float)
        idx_vals = np.array([item[0] for item in rows], dtype=int)
        mask = idx_vals <= frame_idx
        if not np.any(mask):
            continue

        use_ax = ax_p if method in P_VALUE_METHODS else extra_axes[method]
        color = METHOD_COLORS.get(method, None)
        (line,) = use_ax.plot(
            x_vals[mask],
            y_vals[mask],
            marker="o",
            markersize=2.5,
            linewidth=2.4,
            alpha=0.95,
            color=color,
        )
        lines.append(line)
        labels.append(METHOD_LABELS.get(method, method.upper()))

    ax_p.set_title("Metrics vs. Sigma (low means reject null)")
    ax_p.set_xlabel("Sigma (large to small)")
    ax_p.set_ylabel("p-value metrics (PTED/KS/PQM)")
    ax_p.set_ylim(*pvalue_ylim)
    ax_p.grid(alpha=0.2)
    if true_noise is not None:
        line = ax_p.axvline(
            true_noise,
            color="tab:gray",
            linestyle="--",
            linewidth=1.8,
            alpha=0.65,
            label="Moon width",
        )
        lines.append(line)
        labels.append("Moon width")

    if "fld" in extra_axes:
        extra_axes["fld"].set_ylim(fld_ylim[1], fld_ylim[0])
    if "fid" in extra_axes:
        extra_axes["fid"].set_ylim(fid_ylim[1], fid_ylim[0])

    ax_p.set_xscale("log")
    ax_p.set_xlim(float(np.max(sigma_values)), float(np.min(sigma_values)))

    if lines:
        ax_p.legend(lines, labels, loc="center right")

    fig.savefig(frame_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _make_gif(
    png_paths: list[Path],
    gif_path: Path,
    fps: float,
    loop: int,
    frame_step: int,
) -> Path:
    if imageio is None:
        raise ImportError("imageio is required for GIF creation. Install with: pip install imageio")
    if frame_step <= 0:
        raise ValueError("frame_step must be >= 1")

    pattern = re.compile(r"^sigma_frame_(\d+)\.png$")

    def frame_key(path: Path) -> int:
        match = pattern.match(path.name)
        if match:
            return int(match.group(1))
        return 10**12

    selected = sorted(png_paths, key=frame_key)[::frame_step]
    if not selected:
        raise ValueError("No PNG frames selected for GIF creation")

    raw_frames = [imageio.imread(path) for path in selected]

    # Matplotlib layout can introduce off-by-few-pixel output size differences.
    # Crop all frames to the smallest common shape so GIF stacking is robust.
    min_h = min(frame.shape[0] for frame in raw_frames)
    min_w = min(frame.shape[1] for frame in raw_frames)
    frames = [frame[:min_h, :min_w] for frame in raw_frames]

    # base_duration = 1.0 / max(fps, 1e-6)
    # durations = [base_duration] * len(frames)
    # durations[-1] = durations[-1] + 5.0
    for _ in range(int(5 * fps)):
        frames.append(frames[-1])

    # imageio.mimwrite(gif_path, frames, duration=base_duration, loop=loop)
    imageio.mimwrite(gif_path, frames, fps=fps, loop=loop)
    return gif_path


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    if not args.plot_frames and not args.make_gif:
        args.plot_frames = True
        args.make_gif = True

    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    sigma_values_path = output_dir / "sigma_values.npy"
    if not sigma_values_path.exists():
        raise FileNotFoundError(f"Missing sigma values: {sigma_values_path}")
    sigma_values = np.load(sigma_values_path)

    train_data_path = output_dir / "train_data.npy"
    centers_path = output_dir / "base_centers.npy"
    noise_path = output_dir / "base_noise.npy"
    if not train_data_path.exists() or not centers_path.exists() or not noise_path.exists():
        raise FileNotFoundError("Missing one or more mixture artifact arrays")

    train_data = np.load(train_data_path)
    base_centers = np.load(centers_path)
    base_noise = np.load(noise_path)

    metrics_by_method = _load_metrics(output_dir / "sweep_metrics.csv")

    pvalue_vals = []
    fld_vals = []
    fid_vals = []
    for method, rows in metrics_by_method.items():
        values = np.array([item[2] for item in rows], dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue
        if method in P_VALUE_METHODS:
            pvalue_vals.extend(finite_values.tolist())
        elif method == "fld":
            fld_vals.extend(finite_values.tolist())
        elif method == "fid":
            fid_vals.extend(finite_values.tolist())

    pvalue_ylim = (0.0, 1.0)

    def _metric_ylim(values: list[float]) -> tuple[float, float]:
        if not values:
            return (0.0, 1.0)
        low = float(np.min(values))
        high = float(np.max(values))
        if low == high:
            pad = abs(low) * 0.05 + 1e-3
            return (low - pad, high + pad)
        # pad = 0.05 * (high - low)
        return low, high  # (low - pad, high + pad)

    fld_ylim = _metric_ylim(fld_vals)
    fid_ylim = _metric_ylim(fid_vals)

    true_noise = float(config.get("noise", 0.08))
    xx, yy, density = _load_density_grid(output_dir=output_dir, noise=true_noise)

    frames_dir = output_dir / "sweep_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    if args.plot_frames or args.make_gif:
        for idx, sigma in enumerate(sigma_values):
            generated = base_centers + float(sigma) * base_noise
            frame_path = frames_dir / f"sigma_frame_{idx:04d}.png"
            _plot_frame(
                frame_path=frame_path,
                frame_idx=idx,
                sigma=float(sigma),
                train_data=train_data,
                generated=generated,
                xx=xx,
                yy=yy,
                density=density,
                metrics_by_method=metrics_by_method,
                sigma_values=sigma_values,
                pvalue_ylim=pvalue_ylim,
                fld_ylim=fld_ylim,
                fid_ylim=fid_ylim,
                true_noise=true_noise,
            )
            frame_paths.append(frame_path)

        print(f"Wrote {len(frame_paths)} sweep frames to: {frames_dir}")

    if args.make_gif:
        gif_path = _make_gif(
            png_paths=frame_paths,
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
