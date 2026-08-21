from __future__ import annotations

import numpy as np
from scipy.stats import skewnorm
from ..utils import TwoSampleProblem
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def project_to_first_pc(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    y = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
    z = torch.cat([x, y], dim=0)
    mean = z.mean(axis=0, keepdims=True)
    centered = z - mean
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    pc1 = vt[0]
    x1 = (x - mean) @ pc1
    y1 = (y - mean) @ pc1
    return x1.detach().cpu().numpy(), y1.detach().cpu().numpy()


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
    elif deviation == "skew":
        y = skewnorm.rvs(a=severity, size=(n_samples, 1), random_state=rng)
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
