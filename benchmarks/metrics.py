from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.stats import ks_2samp

from pted import pted

from fld.metrics.FLD import FLD
from pqm import pqm_pvalue


from .datasets.gaussian import project_to_first_pc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_samples(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return values[:, None]
    if values.ndim > 2:
        return values.reshape(values.shape[0], -1)
    return values


def ks_pc1_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    x1, y1 = project_to_first_pc(x, y)
    return float(ks_2samp(np.asarray(x1).ravel(), np.asarray(y1).ravel()).pvalue)


def _as_cov_matrix(values: np.ndarray) -> np.ndarray:
    values = _prepare_samples(values)
    covariance = np.cov(values, rowvar=False, bias=False)
    return np.atleast_2d(covariance)


def _split_baseline_samples(
    x: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    x = _prepare_samples(x)
    if len(x) < 2:
        raise ValueError("FLD requires at least 2 baseline samples")

    shuffled = x[rng.permutation(len(x))]

    split_idx = int(np.floor(0.8 * len(shuffled)))
    split_idx = min(max(split_idx, 1), len(shuffled) - 1)
    return shuffled[:split_idx], shuffled[split_idx:]


def fld_score(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    x = _prepare_samples(x)
    y = _prepare_samples(y)
    train_x, test_x = _split_baseline_samples(x, rng=rng)

    train_feat = torch.as_tensor(train_x, dtype=torch.float32)
    test_feat = torch.as_tensor(test_x, dtype=torch.float32)
    gen_feat = torch.as_tensor(y, dtype=torch.float32)

    return float(FLD().compute_metric(train_feat, test_feat, gen_feat))


def fld_two_sample_score(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    return fld_score(x, y, rng=rng)


def fid_score(x: np.ndarray, y: np.ndarray) -> float:
    x = _prepare_samples(x)
    y = _prepare_samples(y)
    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    sx = _as_cov_matrix(x)
    sy = _as_cov_matrix(y)
    cov_prod = sqrtm(sx @ sy)
    if np.iscomplexobj(cov_prod):
        cov_prod = cov_prod.real
    score = np.sum((mx - my) ** 2) + np.trace(sx + sy - 2.0 * cov_prod)
    return float(np.real(score))


def fid_two_sample_score(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    return fid_score(x, y)


def pted_two_sample_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    return float(
        pted(
            torch.tensor(x, device=DEVICE),
            torch.tensor(y, device=DEVICE),
            permutations=permutations,
        )
    )


def pted_two_sample_pvalue_onetail(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    return float(
        pted(
            torch.tensor(x, device=DEVICE),
            torch.tensor(y, device=DEVICE),
            two_tailed=False,
            permutations=permutations,
        )
    )


def pqm_mean_chi2_and_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    x = _prepare_samples(x)
    y = _prepare_samples(y)

    # PQM internally samples references from x/y (and optional Gaussian draws).
    # Ensure `num_refs` cannot exceed sample-size-safe limits for small prototype runs.
    min_samples = min(len(x), len(y))
    if min_samples < 4:
        raise ValueError("PQM requires at least 4 samples per set")
    safe_num_refs = min(int(100), max(4, min_samples - 3))

    pqm_values = pqm_pvalue(
        torch.tensor(x, device=DEVICE),
        torch.tensor(y, device=DEVICE),
        num_refs=safe_num_refs,
        re_tessellation=permutations,
    )

    pqm_values = np.asarray(pqm_values, dtype=float)
    return float(np.median(pqm_values))


def metric_sweep():
    # Default metric map for runners that need comparable sweep scores.
    return {
        "pted": pted_two_sample_pvalue,
        "ks_pc1": ks_pc1_pvalue,
        "fld": fld_two_sample_score,
        "fid": fid_two_sample_score,
        "pqm": pqm_mean_chi2_and_pvalue,
    }
