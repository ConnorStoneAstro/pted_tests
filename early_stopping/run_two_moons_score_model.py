from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from pted import pted
from torch.utils.data import TensorDataset

try:
    from score_models import MLP, ScoreModel
except ImportError as exc:
    raise ImportError(
        "score_models is required for this early stopping test. Install with: pip install score_models"
    ) from exc

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from early_stopping.two_moons import sample_two_moons, two_moons_density


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
        description="Train a score-based diffusion model on two moons and track PTED dynamics"
    )
    parser.add_argument("--output-dir", default="early_stopping/results/two_moons")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-pted-generated", type=int, default=512)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument(
        "--epochs-per-eval",
        type=int,
        default=1000,
        help="Number of training epochs to run before each sampling/PTED evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--sampling-steps", type=int, default=256)
    parser.add_argument("--permutations", type=int, default=512)
    parser.add_argument("--sigma-min", type=float, default=1e-2)
    parser.add_argument("--sigma-max", type=float, default=5.0)
    parser.add_argument("--mlp-layers", type=int, default=4)
    parser.add_argument("--mlp-units", type=int, default=256)
    parser.add_argument("--stop-patience", type=int, default=8)
    parser.add_argument("--stop-tolerance", type=float, default=0.01)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--save-model-checkpoints",
        action="store_true",
        help="Save score_model checkpoints under output_dir/checkpoints for later reuse",
    )
    return parser


def _to_dataset(samples: np.ndarray) -> TensorDataset:
    tensor = torch.from_numpy(samples.astype(np.float32, copy=False))
    return TensorDataset(tensor)


def _train_one_epoch(
    model: ScoreModel,
    dataset: TensorDataset,
    learning_rate: float,
    ema_decay: float,
    batch_size: int,
    checkpoints_directory: Path | None,
    epochs: int,
) -> float:
    if epochs <= 0:
        raise ValueError("epochs must be >= 1")
    losses = model.fit(
        dataset=dataset,
        epochs=epochs,
        learning_rate=learning_rate,
        ema_decay=ema_decay,
        batch_size=batch_size,
        shuffle=True,
        warmup=0,
        clip=0.0,
        checkpoints_directory=(
            str(checkpoints_directory) if checkpoints_directory is not None else None
        ),
        verbose=0,
    )
    return float(np.mean(losses))


def _sample_model(model: ScoreModel, n_samples: int, sampling_steps: int) -> np.ndarray:
    with torch.no_grad():
        generated = model.sample(shape=[n_samples, 2], steps=sampling_steps)
    return generated.detach().cpu().numpy().astype(np.float32)


def _pted_pvalue(
    real_samples: np.ndarray, generated_samples: np.ndarray, permutations: int
) -> float:
    x = torch.from_numpy(real_samples.astype(np.float32, copy=False))
    y = torch.from_numpy(generated_samples.astype(np.float32, copy=False))
    return float(pted(x, y, permutations=permutations))


def _save_reference_density(output_dir: Path, noise: float) -> None:
    xx, yy = np.meshgrid(
        np.linspace(-1.8, 2.8, 220),
        np.linspace(-1.4, 1.8, 220),
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    density = two_moons_density(grid, noise=noise, quadrature_points=512).reshape(xx.shape)
    np.savez_compressed(
        output_dir / "reference_density.npz",
        xx=xx.astype(np.float32),
        yy=yy.astype(np.float32),
        density=density.astype(np.float32),
    )


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "epoch_samples"
    checkpoints_dir = output_dir / "checkpoints" if args.save_model_checkpoints else None
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    if checkpoints_dir is not None:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data = sample_two_moons(args.n_train, rng=rng, noise=args.noise)
    np.save(output_dir / "train_data.npy", train_data)
    _save_reference_density(output_dir=output_dir, noise=args.noise)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    net = MLP(
        dimensions=2,
        layers=args.mlp_layers,
        units=args.mlp_units,
        time_embedding_dimensions=32,
        embedding_scale=32,
    )
    model = ScoreModel(
        model=net,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        device=device,
    )

    history: list[EpochRecord] = []
    best_pvalue = -1.0
    best_epoch = 1
    suggested_stop_epoch = -1
    declines_after_best = 0

    if args.epochs_per_eval <= 0:
        raise ValueError("--epochs-per-eval must be >= 1")

    trained_epochs = 0
    while trained_epochs < args.epochs:
        eval_epoch = min(args.epochs, trained_epochs + args.epochs_per_eval)

        chunk_epochs = eval_epoch - trained_epochs
        batch_size = min(max(1, args.batch_size), len(train_data))
        train_loss = _train_one_epoch(
            model=model,
            dataset=_to_dataset(train_data),
            learning_rate=args.learning_rate,
            ema_decay=args.ema_decay,
            batch_size=batch_size,
            checkpoints_directory=checkpoints_dir,
            epochs=chunk_epochs,
        )
        trained_epochs = eval_epoch

        effective_train_subset_size = len(train_data)

        generated = _sample_model(
            model=model,
            n_samples=args.n_pted_generated,
            sampling_steps=args.sampling_steps,
        )
        pvalue = _pted_pvalue(
            real_samples=train_data,
            generated_samples=generated,
            permutations=args.permutations,
        )

        if pvalue > best_pvalue:
            best_pvalue = pvalue
            best_epoch = eval_epoch
            declines_after_best = 0
        elif best_pvalue - pvalue > args.stop_tolerance:
            declines_after_best += 1

        if suggested_stop_epoch < 0 and declines_after_best >= args.stop_patience:
            suggested_stop_epoch = best_epoch

        if suggested_stop_epoch < 0:
            suggested_stop_epoch_for_record = best_epoch
        else:
            suggested_stop_epoch_for_record = suggested_stop_epoch

        record = EpochRecord(
            epoch=eval_epoch,
            train_loss=train_loss,
            pted_pvalue=pvalue,
            effective_train_subset_size=effective_train_subset_size,
            best_epoch_so_far=best_epoch,
            best_pvalue_so_far=best_pvalue,
            suggested_early_stop_epoch=suggested_stop_epoch_for_record,
        )
        history.append(record)

        np.save(samples_dir / f"epoch_{eval_epoch:03d}.npy", generated)

        print(
            f"epoch={eval_epoch:03d} "
            f"loss={train_loss:.6f} "
            f"pted_pvalue={pvalue:.4f} "
            f"effective_train_subset_size={effective_train_subset_size} "
            f"suggested_stop={suggested_stop_epoch_for_record}"
        )

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(history[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in history)

    print(f"Saved training/validation artifacts to: {output_dir}")
    print(
        "Next step: run `python early_stopping/plot_two_moons_results.py "
        f"--output-dir {output_dir}` to build plots and GIF."
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
