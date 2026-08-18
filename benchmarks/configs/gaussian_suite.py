import numpy as np

DEVIATION_KINDS = ["mean_shift", "scale_shift", "bimodal", "contamination"]

CONFIG = {
    "output_dir": "benchmarks/results/gaussian_suite_onetail",
    "permutations": 512,
    "seeds": list(range(16)),
    "n_samples": 100,
    "deviations": DEVIATION_KINDS,
    "severities": np.linspace(0.0, 1.0, 9).tolist(),
}
