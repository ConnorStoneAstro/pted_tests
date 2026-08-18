import numpy as np

DEVIATION_KINDS = ["white_noise", "pair_blend", "class_drop"]
DATASETS = ["gaussian2x2", "mnist", "cifar10"]

CONFIG = {
    "output_dir": "benchmarks/results/vision_suite",
    "data_root": "benchmarks/data",
    "download": False,
    "permutations": 512,
    "seeds": [0, 1, 2, 3, 4],
    "datasets": ["mnist"],
    "deviations": DEVIATION_KINDS,
    "n_samples": 1024,
    "severities": np.linspace(0.0, 1.0, 9).tolist(),
}
