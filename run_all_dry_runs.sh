#!/usr/bin/env bash
set -e

# Run from the repository root after installing requirements.
# This script checks that the main experiment entry points can parse their
# default configuration without launching the expensive experiment loops.

# Benchmark suite dry runs.
echo "Running benchmark suite dry runs..."
python benchmarks/run_gaussian_suite.py --dry-run
python benchmarks/run_vision_suite.py --dry-run

# Gaussian posterior coverage dry run.
echo "Running Gaussian posterior coverage dry run..."
python coverage_test/run_coverage_test.py --dry-run

# Smooth two-moons mixture dry run.
echo "Running smooth two-moons mixture dry run..."
python early_stopping_smooth_mixture/run_sigma_sweep.py --dry-run
echo "All dry runs completed successfully!"