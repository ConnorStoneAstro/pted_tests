# Coverage Test Workspace

This directory contains a Gaussian posterior coverage sweep. It tests whether posterior samples are calibrated against known ground-truth parameters as the posterior covariance is scaled across a range of sigma values.

## How The Test Works

`run_coverage_test.py` simulates `n_sims` independent Gaussian inference problems. For each problem it draws:

- a ground-truth 2D parameter from a broad Gaussian prior
- one observed data point around that parameter
- posterior samples centered on the observed data point

The script then rescales the posterior samples by each value in a log-spaced sigma grid. A scale near `1.0` represents the nominal posterior width. Smaller values are overconfident and larger values are underconfident.

For each sigma, the script computes coverage-oriented p-values with PTED, MIRA, HPD, MMD, and KS-style comparisons.

## Run The Coverage Sweep

Run from the repository root:

```bash
python coverage_test/run_coverage_test.py
```

To inspect the default configuration without running the sweep:

```bash
python coverage_test/run_coverage_test.py --dry-run
```

The default output directory is `coverage_test/results/coverage_test`.

## Configuration Options

All configuration is provided through command-line flags:

- `--output-dir`: directory where arrays and plots are written
- `--seed`: random seed for the simulated data and posterior samples
- `--n-sims`: number of independent simulated inference problems
- `--n-posterior-samples`: posterior draws per simulated problem
- `--permutations`: permutation count used by the statistical tests
- `--sigma-min`: smallest posterior scale in the sweep
- `--sigma-max`: largest posterior scale in the sweep
- `--n-sigmas`: number of log-spaced sigma values
- `--dry-run`: print the resolved configuration and sigma grid, then exit

A smaller local check looks like:

```bash
python coverage_test/run_coverage_test.py \
  --output-dir coverage_test/results/coverage_quick \
  --n-sims 12 \
  --n-posterior-samples 32 \
  --permutations 20 \
  --sigma-min 0.5 \
  --sigma-max 2.0 \
  --n-sigmas 5
```

## Plotting

After a sweep has written artifacts, run:

```bash
python coverage_test/plot_coverage_test.py
```

To plot a non-default output directory:

```bash
python coverage_test/plot_coverage_test.py \
  --output-dir coverage_test/results/coverage_quick
```

Plotting options are:

- `--output-dir`: directory containing sweep artifacts
- `--gif-name`, `--gif-fps`, and `--gif-loop`: parsed by the current CLI, but the current plotting path writes static PNG summaries

## Expected Outputs

`run_coverage_test.py` writes NumPy arrays including:

- `sigma_values.npy`
- `pvalues_pted.npy`
- `pvalues_mira.npy`
- `pvalues_hdp.npy`
- `pvalues_mmd.npy`
- `pvalues_ks.npy`
- `ground_truth.npy`
- `data_values.npy`
- `data_cov.npy`
- `posterior_samples.npy`

`plot_coverage_test.py` adds summary visualizations, including:

- `coverage_sweep_summary.png`
- `coverage_case_comparison.png`
- `pit_plot.png`
