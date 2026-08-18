from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from ..datasets.vision import generate_vision_problem, load_vision_dataset
from ..metrics import metric_sweep
from time import process_time


def run_vision_sweep(
    datasets: Iterable[str],
    deviations: Iterable[str],
    severities: Iterable[float],
    n_samples: int | None = None,
    seeds: Iterable[int] = (0, 1, 2),
    permutations: int = 200,
    metrics: dict[str, Callable] = metric_sweep(),
) -> list[dict[str, float | int | str | None]]:
    records: list[dict[str, float | int | str | None]] = []
    for dataset_name in datasets:
        print(f"Running vision sweep for dataset: {dataset_name}")
        dataset = load_vision_dataset(dataset_name)
        for deviation in deviations:
            print(f"Running vision sweep for deviation: {deviation}")
            for severity in severities:
                print(f"Running vision sweep for severity: {severity}")
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    problem = generate_vision_problem(
                        dataset=dataset,
                        severity=severity,
                        deviation=deviation,
                        n_samples=n_samples,
                        rng=rng,
                    )
                    sub_record = []
                    for name, metric in metrics.items():
                        print(f"Running vision sweep for metric: {name}")
                        start = process_time()
                        value = metric(problem.x, problem.y, permutations=permutations, rng=rng)
                        runtime = process_time() - start
                        sub_record.append(
                            {
                                "method": name,
                                "dataset": dataset_name,
                                "severity": severity,
                                "deviation": deviation,
                                "seed": seed,
                                "score": value,
                                "runtime": runtime,
                            }
                        )
                    records.extend(sub_record)
    return records
