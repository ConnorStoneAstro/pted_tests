# Smooth Mixture Sigma Sweep (Two Moons)

This directory contains a two-stage experiment for a simple Gaussian mixture
"training" model on two moons.

- Stage 1 (`run_sigma_sweep.py`): evaluate metrics across a sigma sweep.
- Stage 2 (`plot_sigma_sweep.py`): render side-by-side frames and a GIF.

## Experiment Idea

- Build a mixture model with one Gaussian component per training sample.
- Fix one set of component assignments and base Gaussian noise vectors once.
- For each sigma in a sweep from large to small, generate samples as:

  `generated = center[component] + sigma * base_noise`

This keeps samples coupled across sigma values and yields smoother metric curves.

## Metrics

The sweep uses the same metric map as benchmarks via `benchmarks.metrics.metric_sweep()`:

- `pted`
- `ks_pc1`
- `fld`
- `fid`
- `pqm`

## Stage 1: Evaluate Sweep

```bash
python early_stopping_smooth_mixture/run_sigma_sweep.py \
  --output-dir early_stopping_smooth_mixture/results/two_moons_sigma_sweep \
  --n-train 200 \
  --n-generated 1024 \
  --sigma-max 1.0 \
  --sigma-min 0.005 \
  --n-sigmas 120 \
  --sigma-schedule log \
  --permutations 1000
```

Main outputs:

- `run_config.json`
- `train_data.npy`
- `sigma_values.npy`
- `component_indices.npy`
- `base_noise.npy`
- `base_centers.npy`
- `sweep_metrics.csv`
- `reference_density.npz`

## Stage 2: Plot + GIF

```bash
python early_stopping_smooth_mixture/plot_sigma_sweep.py \
  --output-dir early_stopping_smooth_mixture/results/two_moons_sigma_sweep \
  --gif-name sigma_sweep_animation.gif
```

This produces:

- `sweep_frames/sigma_frame_XXXX.png`
- `sigma_sweep_animation.gif`

Each frame contains:

- Left: generated samples overlaid on true two-moons density.
- Right: metric curves drawn only up to the current sigma index, so the lines
  appear to be filled in over the sweep.
