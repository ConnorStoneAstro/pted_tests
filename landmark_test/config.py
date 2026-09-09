import numpy as np

CONFIG = {
    "output_dir": "landmark_test/results/landmark_test_newpted",
    "dataset": "mnist",
    "deviation": "white_noise",
    "n_samples": 8192,
    "n_landmarks": [8192, 2048, 512, 128, 32, 8],
    "permutations": 512,
    "seeds": list(np.arange(64)),
    "severities": np.linspace(0.0, 0.5, 15).tolist(),
    "two_tailed": False,
}
