from __future__ import annotations

from typing import Iterable, Callable

from ..datasets.gaussian import generate_gaussian_problem
from ..metrics import metric_sweep
import numpy as np
from time import process_time


def run_gaussian_sweep(
    deviations: Iterable[str],
    severities: Iterable[float],
    n_samples: int = 100,
    permutations: int = 200,
    seeds: Iterable[int] = (0, 1, 2),
    metrics: dict[str, Callable] = metric_sweep(),
) -> list[dict[str, float | int | str | None]]:
    records: list[dict[str, float | int | str | None]] = []
    for deviation in deviations:
        print("deviation:", deviation)
        for severity in severities:
            print("severity:", severity)
            for seed in seeds:
                rng = np.random.default_rng(seed)
                problem = generate_gaussian_problem(
                    severity=severity,
                    deviation=deviation,
                    n_samples=n_samples,
                    rng=rng,
                )
                sub_record = []
                for name, metric in metrics.items():
                    print("metric:", name)
                    start = process_time()
                    value = metric(problem.x, problem.y, permutations=permutations, rng=rng)
                    runtime = process_time() - start
                    sub_record.append(
                        {
                            "method": name,
                            "severity": severity,
                            "deviation": deviation,
                            "seed": seed,
                            "score": value,
                            "runtime": runtime,
                        }
                    )
                records.extend(sub_record)
    return records
