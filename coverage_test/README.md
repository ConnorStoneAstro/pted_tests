# Coverage Test Workspace

This directory contains a Gaussian posterior coverage sweep. It tests whether posterior samples are calibrated against known ground-truth parameters as the posterior covariance is scaled across a range of sigma values.

## Bootstrap MIRA Coverage Experiment

The separate [bootstrap runner](run_bootstrap_coverage_test.py) uses the same
Gaussian simulation setup as the original experiment, but evaluates only PTED
and MIRA. Defaults are 64 simulations, 128 posterior samples per simulation,
512 PTED permutations / MIRA Monte Carlo runs, and 16 linearly spaced sigma
scales from 0.5 through 2.0 in steps of 0.1.

At each scale it evaluates the original MIRA score and 256 bootstrap datasets.
Each bootstrap independently draws 128 sample indices with replacement for
**each posterior**, preserving whole 2D draws and keeping the ground truths
and generating posterior centers fixed. It does not resample simulations or
recenter bootstrapped draws. Every MIRA evaluation uses a fresh reproducible
Torch seed; the bootstrap score spread therefore includes MIRA Monte Carlo
noise as well as posterior resampling uncertainty.
Simulation generation, bootstrap indices, MIRA seeds, and PTED permutations
use separately seeded streams so the full run is reproducible.

The [bootstrap plotter](plot_bootstrap_coverage_test.py) shows PTED p-values on
the left axis and raw MIRA scores on the right. The right axis also shows the
2/3 reference and an adaptive shaded band `2/3 +/- std`, using the sample
standard deviation (`ddof=1`) of that scale's 256 bootstrap scores. This is
conditional uncertainty for the fixed simulations, not a guaranteed
confidence interval or an estimate of variability across new simulations.
No analytic MIRA variance window is used.

Run from the repository root:

```bash
python coverage_test/run_bootstrap_coverage_test.py --dry-run
python coverage_test/run_bootstrap_coverage_test.py
python coverage_test/plot_bootstrap_coverage_test.py
```

Results are isolated in `coverage_test/results/bootstrap_coverage_test`.
The full default run requires 4,112 MIRA evaluations (each with 512 internal
runs), plus 16 PTED evaluations. A small end-to-end check is:

```bash
python coverage_test/run_bootstrap_coverage_test.py \
  --output-dir coverage_test/results/bootstrap_quick \
  --n-sims 8 --n-posterior-samples 16 --permutations 8 \
  --n-bootstraps 4 --sigma-min 0.5 --sigma-max 2.0 --sigma-step 0.5
python coverage_test/plot_bootstrap_coverage_test.py \
  --output-dir coverage_test/results/bootstrap_quick
```

The runner accepts `--seed`, `--n-sims`, `--n-posterior-samples`,
`--permutations`, `--n-bootstraps`, `--sigma-min`, `--sigma-max`,
`--sigma-step`, `--output-dir`, and `--dry-run`. The sigma range must be an
integer multiple of the positive step. Counts must be at least two.

Saved artifacts include:

- `run_config.json`: configuration, resolved scales, and calibration conventions
- `sigma_values.npy`, `pvalues_pted.npy`, `scores_mira.npy`: one value per scale
  (MIRA scores are not p-values)
- `bootstrap_scores_mira.npy`: shape `(n_sigmas, n_bootstraps)`
- `bootstrap_std_mira.npy`: per-scale sample standard deviation, not standard error
- `mira_seeds.npy`: shape `(n_sigmas, n_bootstraps + 1)`; the first column seeds
  the original score and the remaining columns seed bootstrap evaluations
- `ground_truth.npy`, `data_values.npy`, `data_cov.npy`, `posterior_samples.npy`:
  the same simulation artifacts as the original sweep
- `pit_plot_sigma_*.pdf`: per-scale PTED PIT plots

The plotter adds `bootstrap_coverage_sweep_summary.pdf` and a PNG counterpart.
The original runner and plotter retain their existing experiment behavior.

Run the focused regression tests with:

```bash
python -m unittest coverage_test.test_bootstrap_coverage_test -v
```

## Independent MIRA Null Reruns

The [null runner](run_mira_null_test.py) tests whether the within-posterior
bootstrap spread predicts MIRA variability across newly generated systems.
It uses **MIRA only**, at sigma **1.0**, with 64 simulations and 128 posterior
draws each. It first generates one independent reference system and evaluates
its MIRA score and 256 separately bootstrapped versions. It then generates
256 additional independent systems, each with new ground truths, covariance
matrices, observations, and posterior samples. The reference is **not** one
of these reruns. Every evaluation uses 512 internal MIRA runs and a fresh
reproducible seed.

```bash
python coverage_test/run_mira_null_test.py --dry-run
python coverage_test/run_mira_null_test.py
python coverage_test/plot_mira_null_test.py
```

The default output directory is `coverage_test/results/mira_null_test`.
The experiment requires **513 MIRA evaluations**, rather than bootstrapping
every rerun. All count settings can be overridden with `--n-sims`,
`--n-posterior-samples`, `--n-repeats`, `--n-bootstraps`, and `--mira-runs`.
The runner also accepts `--seed`, `--output-dir`, and `--dry-run`.

The [null plotter](plot_mira_null_test.py) writes a four-panel PDF/PNG with:

- a histogram of independent null scores, the reference bootstrap scores
  **recentered by subtracting their mean and adding 2/3**, and the Gaussian
  prediction `Normal(2/3, bootstrap_std)`
- a Q-Q comparison of `(MIRA - 2/3) / bootstrap_std` to a standard normal,
  without fitting away empirical bias or rescaling the observed spread
- empirical CDFs versus the reference prediction
- observed fractions inside `2/3 +/- {1, 2, 3} * bootstrap_std`, compared with
  Gaussian expectations (approximately 68.27%, 95.45%, and 99.73%)

Normality is a hypothesis, not implied by the variance. The bootstrap estimates
conditional uncertainty for **one** fixed system, whereas these reruns include
between-system variation. A mismatch is a meaningful possible result, not an
error to correct by fitting the empirical null variance. The reference variance
itself is uncertain, and 256 reruns cannot resolve extreme tails accurately.

Saved outputs include `scores_mira.npy` (rerun scores),
`bootstrap_scores_mira.npy` (raw, unrecentered reference bootstrap scores),
`reference_score_mira.npy`, `bootstrap_std_mira.npy` (`ddof=1`), and
`z_scores_mira.npy`. `null_summary.json` records bias, empirical and bootstrap
standard deviations, their variance ratio, and coverage counts/fractions.
`run_config.json` records the experiment conventions. `mira_seeds.npy` stores
seeds in evaluation order: reference, bootstrap evaluations, then reruns.
The reference system is saved as `reference_{ground_truth,data_values,data_cov,posterior_samples}.npy`;
the unprefixed counterparts store all rerun systems with a leading repeat axis.
The plotter writes `mira_null_summary.pdf` and `mira_null_summary.png`.

A quick end-to-end check:

```bash
python coverage_test/run_mira_null_test.py \
  --output-dir coverage_test/results/mira_null_quick \
  --n-sims 8 --n-posterior-samples 16 \
  --n-repeats 8 --n-bootstraps 8 --mira-runs 8
python coverage_test/plot_mira_null_test.py \
  --output-dir coverage_test/results/mira_null_quick
python -m unittest coverage_test.test_mira_null_test -v
```

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
