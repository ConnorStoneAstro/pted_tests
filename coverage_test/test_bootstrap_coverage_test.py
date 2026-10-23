from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import torch

from coverage_test.gaussian_posteriors import simulate_posteriors
from coverage_test.plot_bootstrap_coverage_test import plot_sweep_summary
from coverage_test.run_bootstrap_coverage_test import (
    bootstrap_posteriors,
    build_arg_parser,
    build_sigma_values,
    main,
    run_bootstrap_sweep,
)


class BootstrapCoverageTests(unittest.TestCase):
    def test_default_configuration_and_grid(self) -> None:
        args = build_arg_parser().parse_args([])
        self.assertEqual((args.n_sims, args.n_posterior_samples, args.n_bootstraps), (64, 128, 256))
        self.assertEqual(args.permutations, 512)
        np.testing.assert_allclose(
            build_sigma_values(args.sigma_min, args.sigma_max, args.sigma_step),
            np.arange(5, 21) / 10,
            rtol=0,
            atol=1e-15,
        )

    def test_invalid_sigma_grids(self) -> None:
        for limits in (
            (0, 2, 0.1), (2, 0.5, 0.1), (0.5, 2, 0),
            (0.5, 2, -0.1), (0.5, 2, 0.4), (0.5, np.inf, 0.1),
            (np.nan, 2, 0.1),
        ):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                build_sigma_values(*limits)

    def test_invalid_counts_are_cli_errors(self) -> None:
        for flag in ("--n-sims", "--n-posterior-samples", "--n-bootstraps", "--permutations"):
            with self.subTest(flag=flag):
                with patch("sys.argv", ["runner", flag, "1", "--dry-run"]):
                    with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                        main()
                self.assertEqual(error.exception.code, 2)

    def test_bootstrap_preserves_simulations_and_whole_draws(self) -> None:
        samples = np.empty((128, 64, 2))
        samples[:, :, 0] = np.arange(128)[:, None] + 1000 * np.arange(64)[None, :]
        samples[:, :, 1] = -samples[:, :, 0]
        result = bootstrap_posteriors(samples, np.random.default_rng(42))
        indices = np.random.default_rng(42).integers(128, size=(128, 64))
        self.assertEqual(result.shape, samples.shape)
        np.testing.assert_array_equal(result, samples[indices, np.arange(64)[None, :], :])
        np.testing.assert_array_equal(result[:, :, 1], -result[:, :, 0])
        np.testing.assert_array_equal(result[:, :, 0] // 1000, np.broadcast_to(np.arange(64), (128, 64)))
        self.assertFalse(np.array_equal(indices[:, 0], indices[:, 1]))
        self.assertLess(len(np.unique(indices[:, 0])), 128)

    def test_simulation_extraction_preserves_original_rng_sequence(self) -> None:
        rng = np.random.default_rng(42)
        truth = rng.multivariate_normal(np.zeros(2), np.diag([10.0, 10.0]), size=64)
        covariances, centers = [], []
        for gt in truth:
            sigma = rng.uniform(0.5, 2.0, size=2)
            cov = rng.uniform(-1, 1) * np.prod(sigma)
            dc = np.array([[sigma[0] ** 2, cov], [cov, sigma[1] ** 2]], dtype=np.float64)
            covariances.append(dc)
            centers.append(rng.multivariate_normal(mean=gt, cov=dc))
        samples = np.stack(
            [rng.multivariate_normal(dv, dc, size=128) for dv, dc in zip(centers, covariances)],
            axis=1,
        )
        expected = (truth, np.stack(centers), np.stack(covariances), samples)
        actual_rng = np.random.default_rng(42)
        actual = simulate_posteriors(actual_rng, 64, 128)
        for original, extracted in zip(expected, actual):
            np.testing.assert_array_equal(original, extracted)
        self.assertEqual(rng.random(), actual_rng.random())

    def test_default_sweep_artifacts_seeds_and_plot(self) -> None:
        seen_seeds = []

        def fake_mira(truth, posterior, *, num_runs, norm, disable_tqdm):
            self.assertEqual(tuple(truth.shape), (64, 2))
            self.assertEqual(tuple(posterior.shape), (1, 64, 128, 2))
            self.assertEqual(num_runs, 512)
            self.assertTrue(norm)
            self.assertTrue(disable_tqdm)
            seen_seeds.append(torch.initial_seed())
            return torch.rand(1), torch.zeros(1)

        module = "coverage_test.run_bootstrap_coverage_test"
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            args = build_arg_parser().parse_args(["--output-dir", directory])
            with (
                patch(f"{module}.mira", side_effect=fake_mira),
                patch(f"{module}.pted_coverage_test", return_value=0.5) as pted,
                patch("builtins.print"),
            ):
                run_bootstrap_sweep(args)
                self.assertEqual(pted.call_count, 16)
                for call in pted.call_args_list:
                    self.assertEqual(tuple(call.args[1].shape), (128, 64, 2))
                    self.assertIsInstance(call.kwargs["rng"], np.random.Generator)
            seeds = np.load(output_dir / "mira_seeds.npy")
            scores = np.load(output_dir / "bootstrap_scores_mira.npy")
            original_scores = np.load(output_dir / "scores_mira.npy")
            std = np.load(output_dir / "bootstrap_std_mira.npy")
            self.assertEqual(scores.shape, (16, 256))
            self.assertEqual(original_scores.shape, (16,))
            self.assertEqual(seeds.shape, (16, 257))
            self.assertEqual(len(set(seen_seeds)), 16 * 257)
            np.testing.assert_array_equal(seeds.ravel(), seen_seeds)
            np.testing.assert_array_equal(std, scores.std(axis=1, ddof=1))
            np.testing.assert_array_equal(np.load(output_dir / "pvalues_pted.npy"), np.full(16, 0.5))
            config = json.loads((output_dir / "run_config.json").read_text())
            self.assertEqual(config["bootstrap_std_ddof"], 1)
            self.assertEqual(config["window_std_multiplier"], 1)
            self.assertEqual(len(config["sigma_values"]), 16)

            with patch("coverage_test.plot_bootstrap_coverage_test.plt.close"):
                path = plot_sweep_summary(output_dir)
            fig = plt.gcf()
            self.addCleanup(plt.close, fig)
            self.assertEqual(len(fig.axes), 2)
            ax, ax_mira = fig.axes
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), np.full(16, 0.5))
            np.testing.assert_array_equal(ax_mira.lines[0].get_ydata(), original_scores)
            np.testing.assert_array_equal(ax_mira.lines[1].get_ydata(), [2 / 3, 2 / 3])
            vertices = ax_mira.collections[0].get_paths()[0].vertices
            sigma_values = np.load(output_dir / "sigma_values.npy")
            for sigma, width in zip(sigma_values, std):
                band_values = vertices[vertices[:, 0] == sigma, 1]
                self.assertAlmostEqual(band_values.min(), 2 / 3 - width)
                self.assertAlmostEqual(band_values.max(), 2 / 3 + width)
            self.assertGreater(path.stat().st_size, 0)
            self.assertGreater(path.with_suffix(".png").stat().st_size, 0)

    def test_reproducibility(self) -> None:
        def fake_mira(*_args, **_kwargs):
            return torch.rand(1), torch.zeros(1)

        module = "coverage_test.run_bootstrap_coverage_test"
        with tempfile.TemporaryDirectory() as directory:
            outputs = [Path(directory) / "first", Path(directory) / "second"]
            with (
                patch(f"{module}.mira", side_effect=fake_mira),
                patch(f"{module}.pted_coverage_test", return_value=0.5),
                patch("builtins.print"),
            ):
                for output in outputs:
                    args = build_arg_parser().parse_args([
                        "--output-dir", str(output), "--n-sims", "4",
                        "--n-posterior-samples", "8", "--n-bootstraps", "4",
                        "--permutations", "4", "--sigma-step", "0.5",
                    ])
                    run_bootstrap_sweep(args)
            for artifact in outputs[0].glob("*.npy"):
                np.testing.assert_array_equal(
                    np.load(artifact), np.load(outputs[1] / artifact.name)
                )

    def test_real_metric_reproducibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = [Path(directory) / "first", Path(directory) / "second"]
            for output in outputs:
                args = build_arg_parser().parse_args([
                    "--output-dir", str(output), "--n-sims", "4",
                    "--n-posterior-samples", "8", "--n-bootstraps", "4",
                    "--permutations", "4", "--sigma-min", "1", "--sigma-max", "1",
                ])
                with patch("builtins.print"):
                    run_bootstrap_sweep(args)
            for artifact in outputs[0].glob("*.npy"):
                np.testing.assert_array_equal(
                    np.load(artifact), np.load(outputs[1] / artifact.name)
                )


if __name__ == "__main__":
    unittest.main()
