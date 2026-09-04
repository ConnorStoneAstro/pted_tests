import numpy as np

CONFIG = {
    "output_dir": "chunk_test/results/chunk_test",
    "dataset": "mnist",
    "deviation": "white_noise",
    "n_samples": 8192,
    "chunk_sizes": [8192, 2048, 512, 128, 32, 8],
    "permutations": 512,
    "seeds": list(np.arange(32)),
    "severities": np.linspace(0.0, 0.5, 15).tolist(),
    "two_tailed": False,
}
