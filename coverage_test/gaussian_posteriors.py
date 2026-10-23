from __future__ import annotations

import numpy as np


def sample_data_covariance(rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(0.5, 2.0, size=2)
    cov = rng.uniform(-1, 1) * np.prod(sigma)
    return np.array([[sigma[0] ** 2, cov], [cov, sigma[1] ** 2]]).astype(np.float64)


def simulate_posteriors(
    rng: np.random.Generator, n_sims: int, n_posterior_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_param_cov = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float64)
    ground_truth = rng.multivariate_normal(mean=np.zeros(2), cov=true_param_cov, size=n_sims)
    data_values = []
    data_cov = []
    for sim_idx in range(n_sims):
        data_cov.append(sample_data_covariance(rng))
        data_values.append(
            rng.multivariate_normal(mean=ground_truth[sim_idx], cov=data_cov[sim_idx])
        )
    data_cov = np.stack(data_cov, axis=0)
    data_values = np.stack(data_values, axis=0)
    posterior_samples = np.stack(
        [
            rng.multivariate_normal(mean=dv, cov=dc, size=n_posterior_samples)
            for dv, dc in zip(data_values, data_cov)
        ],
        axis=1,
    )  # (Nsamp, Nsim, 2)
    return ground_truth, data_values, data_cov, posterior_samples
