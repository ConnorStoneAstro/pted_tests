from __future__ import annotations

import numpy as np


def _moon_curve(theta: np.ndarray, moon_index: int) -> np.ndarray:
    if moon_index == 0:
        x = np.cos(theta)
        y = np.sin(theta)
        return np.stack([x, y], axis=-1)
    if moon_index == 1:
        x = 1.0 - np.cos(theta)
        y = 1.0 - np.sin(theta) - 0.5
        return np.stack([x, y], axis=-1)
    raise ValueError(f"moon_index must be 0 or 1, got {moon_index}")


def sample_two_moons(
    n_samples: int,
    rng: np.random.Generator,
    noise: float = 0.08,
) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    moon = rng.integers(0, 2, size=n_samples)
    theta = rng.uniform(0.0, np.pi, size=n_samples)

    points = np.empty((n_samples, 2), dtype=np.float64)
    points[moon == 0] = _moon_curve(theta[moon == 0], moon_index=0)
    points[moon == 1] = _moon_curve(theta[moon == 1], moon_index=1)

    if noise > 0.0:
        points = points + rng.normal(loc=0.0, scale=noise, size=points.shape)

    return points.astype(np.float32)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - max_values)
    return np.squeeze(max_values, axis=axis) + np.log(np.sum(shifted, axis=axis))


def _component_log_density(
    points: np.ndarray,
    moon_index: int,
    noise: float,
    quadrature_points: int,
) -> np.ndarray:
    theta = np.linspace(0.0, np.pi, quadrature_points, endpoint=False, dtype=np.float64)
    curve = _moon_curve(theta, moon_index=moon_index)

    diff = points[:, None, :] - curve[None, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)

    var = float(noise * noise)
    if var <= 0.0:
        raise ValueError("noise must be positive for density evaluation")

    log_gaussian = -0.5 * sq_dist / var - np.log(2.0 * np.pi * var)
    delta_theta = np.pi / float(quadrature_points)
    return _logsumexp(log_gaussian, axis=1) + np.log(delta_theta) - np.log(np.pi)


def log_two_moons_density(
    points: np.ndarray,
    noise: float = 0.08,
    quadrature_points: int = 1024,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if quadrature_points < 8:
        raise ValueError("quadrature_points must be >= 8")

    log_p0 = _component_log_density(
        points=points,
        moon_index=0,
        noise=noise,
        quadrature_points=quadrature_points,
    )
    log_p1 = _component_log_density(
        points=points,
        moon_index=1,
        noise=noise,
        quadrature_points=quadrature_points,
    )

    stacked = np.stack([log_p0, log_p1], axis=1)
    return _logsumexp(stacked, axis=1) - np.log(2.0)


def two_moons_density(
    points: np.ndarray,
    noise: float = 0.08,
    quadrature_points: int = 1024,
) -> np.ndarray:
    return np.exp(
        log_two_moons_density(
            points=points,
            noise=noise,
            quadrature_points=quadrature_points,
        )
    )
