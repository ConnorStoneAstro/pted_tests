from __future__ import annotations

from pathlib import Path

from typing import Literal
import numpy as np
from ..utils import TwoSampleProblem

VISION_DATASETS = ["mnist", "cifar10", "gaussian2x2"]
VISION_DEVIATIONS = ["pair_blend", "class_drop", "white_noise"]


def generate_vision_problem(
    dataset: tuple[np.ndarray, np.ndarray],
    severity: float,
    deviation: str,
    n_samples: int,
    rng: np.random.Generator,
) -> TwoSampleProblem:
    permute = rng.permutation(len(dataset[0]))
    x = dataset[0][permute[:n_samples]]

    if deviation == "pair_blend":
        y1 = dataset[0][permute[n_samples : 2 * n_samples]]
        y2 = dataset[0][permute[2 * n_samples : 3 * n_samples]]
        y = (1.0 - severity / 2) * y1 + (severity / 2) * y2
    elif deviation == "class_drop":
        labels = dataset[1][permute[n_samples:]]
        classes = np.unique(labels)
        if len(classes) < 2:
            raise ValueError("Dataset must contain at least two classes for class_drop deviation")
        p = np.ones(len(labels))
        p[labels == classes[0]] = 1 - severity
        y = dataset[0][
            rng.choice(permute[n_samples:], size=n_samples, p=p / p.sum(), replace=False)
        ]
    elif deviation == "white_noise":
        y = dataset[0][permute[n_samples : 2 * n_samples]]
        noise = rng.normal(loc=0.0, scale=severity / 3, size=y.shape)
        y = y + noise
    else:
        raise ValueError(f"Unknown deviation kind: {deviation}")

    return TwoSampleProblem(
        x=x.astype(np.float32), y=y.astype(np.float32), severity=severity, deviation=deviation
    )


def load_vision_dataset(
    dataset: Literal["mnist", "cifar10", "gaussian2x2"],
    data_root: Path | None = None,
    download: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for MNIST/CIFAR10 loaders. "
            "Install with: pip install torchvision"
        ) from exc

    dataset_name = str(dataset).strip().lower()
    resolved_root = Path("benchmarks/data") if data_root is None else Path(data_root)

    if dataset_name == "mnist":
        train = datasets.MNIST(root=str(resolved_root), train=True, download=download)
        test = datasets.MNIST(root=str(resolved_root), train=False, download=download)
        images = np.concatenate([train.data.numpy(), test.data.numpy()], axis=0).astype(np.float32)
        labels = np.concatenate([train.targets.numpy(), test.targets.numpy()], axis=0).astype(
            np.int64
        )
        images = images / 255.0
        images = images[..., None]
        return images, labels

    if dataset_name == "cifar10":
        train = datasets.CIFAR10(root=str(resolved_root), train=True, download=download)
        test = datasets.CIFAR10(root=str(resolved_root), train=False, download=download)
        images = np.concatenate([train.data, test.data], axis=0).astype(np.float32)
        labels = np.array(train.targets + test.targets, dtype=np.int64)
        images = images / 255.0
        return images, labels

    if dataset_name == "gaussian2x2":
        A = 10_000
        B = 20_000
        C = 30_000
        images_A = np.random.multivariate_normal(
            mean=np.zeros(4, dtype=float),
            cov=np.eye(4),
            size=A,
        ).reshape(A, 2, 2)
        images_B = np.random.multivariate_normal(
            mean=[0.0, 1.0, 1.0, 0.0],
            cov=np.eye(4) * 0.5,
            size=B,
        ).reshape(B, 2, 2)
        images_C = np.random.multivariate_normal(
            mean=[0.0, 1.0, 2.0, 3.0],
            cov=np.eye(4),
            size=C,
        ).reshape(C, 2, 2)
        images = np.concatenate([images_A, images_B, images_C], axis=0)
        labels = np.array([0] * A + [1] * B + [2] * C, dtype=np.int64)
        return images, labels

    raise ValueError(f"Unsupported dataset: {dataset}")
