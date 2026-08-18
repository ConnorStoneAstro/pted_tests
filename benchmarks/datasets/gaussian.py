from __future__ import annotations

import numpy as np
from ..utils import TwoSampleProblem


def project_to_first_pc(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    z = np.concatenate([x, y], axis=0)
    mean = z.mean(axis=0, keepdims=True)
    centered = z - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = vt[0]
    x1 = (x - mean) @ pc1
    y1 = (y - mean) @ pc1
    return x1, y1


def generate_gaussian_problem(
    severity: float,
    deviation: str,
    n_samples: int,
    rng: np.random.Generator,
) -> TwoSampleProblem:
    x = rng.normal(size=(n_samples, 1))

    if deviation == "mean_shift":
        y = rng.normal(loc=severity, size=(n_samples, 1))
    elif deviation == "scale_shift":
        y = rng.normal(scale=1.0 + severity, size=(n_samples, 1))
    elif deviation == "bimodal":
        mix = rng.uniform(size=n_samples) < 0.5
        y = rng.normal(size=(n_samples, 1))
        y[mix, 0] -= severity
        y[~mix, 0] += severity
    elif deviation == "contamination":
        y = rng.normal(size=(n_samples, 1))
        contam = rng.uniform(size=n_samples) < severity
        if contam.any():
            y[contam] = rng.normal(
                loc=4.0,
                scale=1.0,
                size=(int(contam.sum()), 1),
            )
    else:
        raise ValueError(f"Unknown deviation kind: {deviation}")

    return TwoSampleProblem(x=x, y=y, severity=severity, deviation=deviation)
