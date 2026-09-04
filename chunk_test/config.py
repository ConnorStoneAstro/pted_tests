import numpy as np

CONFIG = {
    "output_dir": "chunk_test/results/chunk_test",
    "dataset": "mnist",
    "deviation": "white_noise",
    "n_samples": 1024,
    "chunk_sizes": [1024, 512, 256, 128, 64, 32],
    "permutations": 512,
    "seeds": [0, 1, 2, 3, 4],
    "severities": np.linspace(0.0, 1.0, 9).tolist(),
    "two_tailed": False,
}
