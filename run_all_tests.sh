#!/usr/bin/env bash
set -e

# Run from the repository root after installing requirements.
# This is a high-level documentation runner that uses each script's defaults.
# These commands can take a long time; see the README in each subfolder for
# dry-run checks and smaller example configurations.

# Benchmark suites.
echo "Running benchmark suites..."
python benchmarks/run_gaussian_suite.py
python benchmarks/run_vision_suite.py

# Gaussian posterior coverage sweep and plots.
echo "Running Gaussian posterior coverage sweep and plots..."
python coverage_test/run_coverage_test.py
python coverage_test/plot_coverage_test.py

# Smooth two-moons mixture sigma sweep and animation.
echo "Running smooth two-moons mixture sigma sweep and animation..."
python early_stopping_smooth_mixture/run_sigma_sweep.py
python early_stopping_smooth_mixture/plot_sigma_sweep.py
echo "All tests completed successfully!"