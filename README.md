# PTED Test Workspaces

This repository collects experiment scripts for exercising PTED and related two-sample or coverage diagnostics. The tests are organized as independent workspaces so each folder can be run on its own with a small local configuration.

## Environment Setup

From the repository root, make a new environment then do:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The vision benchmark suite can use MNIST and CIFAR10 from `benchmarks/data`. To download both datasets as part of setup, run:

```bash
python benchmarks/load_vision_datasets.py --download
```

To download or check only one dataset, pass `--datasets`:

```bash
python benchmarks/load_vision_datasets.py --download --datasets mnist
python benchmarks/load_vision_datasets.py --download --datasets cifar10
```

Run scripts from the repository root so local imports such as `benchmarks.*` resolve correctly.

## Available Test Groups

| Folder | Purpose | Main entry points |
| --- | --- | --- |
| `benchmarks/` | Two-sample benchmark sweeps on synthetic Gaussian and image-like problems. | `python benchmarks/run_gaussian_suite.py`, `python benchmarks/run_vision_suite.py` |
| `coverage_test/` | Posterior coverage sensitivity sweep for Gaussian simulations. | `python coverage_test/run_coverage_test.py`, `python coverage_test/plot_coverage_test.py` |
| `early_stopping_smooth_mixture/` | Smooth sigma sweep for a two-moons Gaussian mixture generator. | `python early_stopping_smooth_mixture/run_sigma_sweep.py`, `python early_stopping_smooth_mixture/plot_sigma_sweep.py` |

Each subfolder has its own README with experiment details, configuration knobs, and expected outputs.

## Run Everything Sequentially

The commented shell script [run_all_tests.sh](run_all_tests.sh) is a high-level reference runner. It lists the default commands for each test group in sequence:

```bash
bash run_all_tests.sh
```

Those default commands can take a long time. For a quick wiring check, use the dry-run examples in each subfolder README, such as:

```bash
python benchmarks/run_gaussian_suite.py --dry-run
python coverage_test/run_coverage_test.py --dry-run
python early_stopping_smooth_mixture/run_sigma_sweep.py --dry-run
```

Use the individual subfolder commands when you want to change sample counts, seeds, permutations, sigma sweeps, datasets, or output directories.