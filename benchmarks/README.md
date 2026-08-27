# Benchmark Workspace

This directory contains two-sample benchmark sweeps for PTED and related metrics. The benchmark scripts compare a reference sample `x` against a perturbed sample `y` while increasing a severity value, then write CSV records and plots showing how each method responds.

## Layout

- `benchmarks/datasets/gaussian.py` contains the synthetic 1D Gaussian benchmark family.
- `benchmarks/datasets/vision.py` contains the lightweight 2x2 Gaussian image prototype plus MNIST and CIFAR10 layouts.
- `benchmarks/metrics.py` contains comparison methods that return sweep scores.
- `benchmarks/runners/gaussian.py` runs synthetic Gaussian sweeps.
- `benchmarks/runners/vision.py` defines the 2x2 Gaussian, MNIST/CIFAR class-drop, white-noise, and pair-blend sweep paths.
- `benchmarks/plots.py` contains shared plotting helpers.
- `benchmarks/configs/gaussian_suite.py` and `benchmarks/configs/vision_suite.py` hold default Python configuration dictionaries.

## Current Status

- Synthetic Gaussian sweeps are runnable now.
- The 2x2 Gaussian image prototype is runnable on a laptop and is intended for quick end-to-end checks.
- KS is evaluated on the first principal component projection.
- FLD and FID are stored as raw scores and normalized during plotting so the null case maps to 1.
- MNIST and CIFAR10 are not downloaded automatically; only the layout for class-split experiments is in place.

## How The Tests Work

Gaussian benchmarks draw `x` from a 1D standard normal distribution and draw `y` from a severity-controlled alternative. Available Gaussian deviations are:

- `mean_shift`
- `scale_shift`
- `bimodal`
- `skew`
- `contamination`

Vision benchmarks flatten image-like samples and run the same metric map. Available vision datasets are:

- `gaussian2x2`, a synthetic 2 by 2 image problem that is available without downloads.
- `mnist`, using local MNIST data.
- `cifar10`, using local CIFAR-10 data.

Available vision deviations are:

- `white_noise`, which adds severity-scaled Gaussian noise.
- `pair_blend`, which blends randomly paired samples.
- `class_drop`, which shifts class balance in one sample.

The class-based vision experiments use a progressive class-mix schedule:

- severity `0.0` means both samples are mixed identically, which is the null case.
- severity near `1.0` means one sample is dominated by `class_a` while the other is dominated by `class_b`.
- the same sweep shape will be used for MNIST and CIFAR10 so their curves can be compared consistently.

The pair-blend contamination test samples `x` directly from the dataset pool and builds `y` by selecting random pairs `(A, B)` and mixing them as `y = (1 - s) * A + s * B`.

## Plotting

The plotting helpers in `benchmarks/plots.py` are intended to support two kinds of figures:

- sweep curves that show how each method responds as severity increases
- small prototype figures that show the 2x2 image means and their difference directly

These are kept outside the core PTED package so they can evolve into a separate analysis workflow later.

The explanatory visualizations in `benchmarks/visualizations.py` show what each configured Gaussian and vision deviation does to the samples. They read the same Python config files as the benchmark runners so the severity grids match the experiment settings.

To generate the default visualization set:

```bash
python benchmarks/visualizations.py
```

## Running Benchmarks

Run from the repository root.

```bash
python benchmarks/run_gaussian_suite.py
python benchmarks/run_vision_suite.py
```

The suite scripts use these default configs:

- Gaussian: `benchmarks/configs/gaussian_suite.py`
- Vision: `benchmarks/configs/vision_suite.py`

To check resolved configuration without running the expensive loops:

```bash
python benchmarks/run_gaussian_suite.py --dry-run
python benchmarks/run_vision_suite.py --dry-run
```

To use a custom config, copy one of the config files and pass it with `--config`:

```bash
python benchmarks/run_gaussian_suite.py --config benchmarks/configs/gaussian_suite.py
python benchmarks/run_vision_suite.py --config benchmarks/configs/vision_suite.py
```

To regenerate plots from existing CSV records:

```bash
python benchmarks/run_gaussian_suite.py --plot-only
python benchmarks/run_vision_suite.py --plot-only
```

Use `--records-csv <path>` with `--plot-only` to plot a non-default CSV.

## Configuration Examples

Gaussian configs are Python files containing a `CONFIG` dictionary. The main options are `output_dir`, `permutations`, `seeds`, `n_samples`, `deviations`, and `severities`.

```python
import numpy as np

CONFIG = {
	"output_dir": "benchmarks/results/gaussian_suite_onetail",
	"permutations": 512,
	"seeds": list(range(64)),
	"n_samples": 100,
	"deviations": ["mean_shift", "scale_shift", "bimodal", "skew", "contamination"],
	"severities": np.linspace(0.0, 1.0, 15).tolist(),
}
```

For a faster local Gaussian run, reduce the grid:

```python
CONFIG = {
	"output_dir": "benchmarks/results/gaussian_quick",
	"permutations": 50,
	"seeds": [0],
	"n_samples": 32,
	"deviations": ["mean_shift", "scale_shift"],
	"severities": [0.0, 0.5, 1.0],
}
```

Vision configs use the same pattern. The active runner reads `output_dir`, `permutations`, `seeds`, `datasets`, `deviations`, `n_samples`, and `severities`.

```python
import numpy as np

CONFIG = {
	"output_dir": "benchmarks/results/vision_suite",
	"data_root": "benchmarks/data",
	"download": False,
	"permutations": 512,
	"seeds": [0, 1, 2, 3, 4],
	"datasets": ["mnist"],
	"deviations": ["white_noise", "pair_blend", "class_drop"],
	"n_samples": 1024,
	"severities": np.linspace(0.0, 1.0, 9).tolist(),
}
```

For a quick run that does not require downloaded vision data, use the synthetic dataset:

```python
CONFIG = {
	"output_dir": "benchmarks/results/vision_quick",
	"permutations": 50,
	"seeds": [0],
	"datasets": ["gaussian2x2"],
	"deviations": ["white_noise", "pair_blend"],
	"n_samples": 32,
	"severities": [0.0, 0.5, 1.0],
}
```

`benchmarks/run_vision_suite.py` orchestrates:

- 2x2 Gaussian image sweep
- MNIST/CIFAR10 class-split sweep
- MNIST/CIFAR10 pair-blend contamination sweep

Use `benchmarks/load_vision_datasets.py --data-root <path>` to load MNIST and CIFAR10 from a directory and confirm the data is available before running the suite.

## Expected Outputs

Gaussian runs write:

- `<output_dir>/gaussian_1d/gaussian_1d_records.csv`
- `<output_dir>/gaussian_1d/gaussian_1d_<deviation>_score.png`

Vision runs write:

- `<output_dir>/vision/vision_suite_records.csv`
- `<output_dir>/vision/vision_<dataset>_<deviation>_score.png`

## Record Schema

Records are now intentionally minimal and score-centric:

- `method`
- `severity`
- task metadata fields (for example `deviation`, `seed`, `dataset`, `test_case`)
- `score`
- `raw_score`