import numpy as np

DEVIATION_KINDS = ["white_noise", "pair_blend", "class_drop"]
DATASETS = ["gaussian2x2", "mnist", "cifar10"]

CONFIG = {
    "output_dir": "benchmarks/results/vision_suite_newpted",
    "data_root": "benchmarks/data",
    "download": False,
    "permutations": 512,
    "seeds": list(range(64)),
    "datasets": ["mnist", "cifar10"],
    "deviations": DEVIATION_KINDS,
    "n_samples": 2048,
    "severities": np.linspace(0.0, 1.0, 15).tolist(),
}
