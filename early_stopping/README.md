# Early Stopping Suite (Two Moons)

This directory contains a standalone experiment that trains a score-based diffusion
model on a low-$N$ two-moons dataset and tracks PTED at every epoch.

The workflow is intentionally split into two stages:

1. training + PTED evaluation artifact generation
2. plotting + GIF rendering from saved artifacts

The workflow is designed to show three phases clearly:

1. underfitting (low PTED p-value)
2. best fit (higher PTED p-value)
3. overfitting (p-value declines again)

To force an overfitting regime for demonstration, training switches from the full
training set to a tiny fixed subset after a configured epoch.

## Files

- `two_moons.py`: two-moons sampler and density evaluator for contour plots.
- `run_two_moons_score_model.py`: train loop using `score_models`, PTED validation,
  and artifact export (`history.csv`, samples, data snapshots).
- `plot_two_moons_results.py`: renders summary/epoch plots and builds a GIF from epoch plots.

## Run

1) Generate artifacts (single long run)

```bash
python early_stopping/run_two_moons_score_model.py
```

Useful overrides:

```bash
python early_stopping/run_two_moons_score_model.py \
  --n-train 100 \
  --epochs 120 \
  --overfit-start-epoch 45 \
  --overfit-subset-size 16 \
  --n-val-generated 512 \
  --permutations 200
```

2) Render plots and GIF from saved artifacts

```bash
python early_stopping/plot_two_moons_results.py --output-dir early_stopping/results/two_moons
```

Render-only examples:

```bash
# Just remake summary and epoch plots
python early_stopping/plot_two_moons_results.py \
  --output-dir early_stopping/results/two_moons \
  --plot-summary --plot-epochs

# Just remake GIF from existing epoch plots (every 2nd frame)
python early_stopping/plot_two_moons_results.py \
  --output-dir early_stopping/results/two_moons \
  --make-gif --frame-step 2 --gif-fps 8
```

## Outputs

The training script writes to `early_stopping/results/two_moons` by default:

- `run_config.json`: exact run configuration
- `train_data.npy`: fixed train sample used for the run
- `val_real_data.npy`: fixed validation real sample used for PTED
- `history.csv`: epoch-wise loss and PTED trend
- `epoch_samples/epoch_XXX.npy`: generated samples for each epoch
- `reference_density.npz`: precomputed true two-moons density grid for plotting

The plotting script writes:

- `pted_summary.png`: PTED and loss curves
- `epoch_plots/epoch_XXX.png`: true density + generated samples each epoch
- `epoch_animation.gif`: GIF from epoch plots
