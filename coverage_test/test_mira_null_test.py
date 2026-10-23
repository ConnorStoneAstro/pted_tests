from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from coverage_test.gaussian_posteriors import simulate_posteriors
from coverage_test.plot_mira_null_test import plot_null_summary
from coverage_test.run_mira_null_test import (
    build_arg_parser,
    main,
    run_null_test,
    summarize_null_scores,
)


class MiraNullTests(unittest.TestCase):
    def test_defaults(self) -> None:
        args = build_arg_parser().parse_args([])
        self.assertEqual(
            (args.n_sims, args.n_posterior_samples, args.n_repeats, args.n_bootstraps, args.mira_runs),
            (64, 128, 256, 256, 512),
        )

    def test_summary_uses_reference_std_and_fixed_threshold(self) -> None:
        bootstrap = np.array([0.4, 0.6, 0.8])
        width = bootstrap.std(ddof=1)
        z = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        scores = 2 / 3 + width * z
        summary = summarize_null_scores(scores, bootstrap)
        self.assertAlmostEqual(summary["bootstrap_std"], width)
        self.assertAlmostEqual(summary["z_mean"], 0)
        self.assertAlmostEqual(summary["z_std"], z.std(ddof=1))
        self.assertAlmostEqual(
            summary["null_to_bootstrap_variance_ratio"],
            scores.var(ddof=1) / bootstrap.var(ddof=1),
        )
        for item, count in zip(summary["coverage"], (2, 4, 6)):
            self.assertEqual(item["observed_count"], count)
            self.assertEqual(item["observed_fraction"], count / 6)
            k = item["std_multiplier"]
            self.assertAlmostEqual(item["gaussian_expected_fraction"], norm.cdf(k) - norm.cdf(-k))
        shifted = summarize_null_scores(scores + 0.1, bootstrap)
        self.assertAlmostEqual(shifted["z_mean"], 0.1 / width)

    def test_invalid_scores_fail_explicitly(self) -> None:
        valid = np.array([0.6, 0.7, 0.8])
        for invalid in (np.array([0.5]), np.array([np.nan, 0.7]), np.array([[0.5, 0.6]])):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                summarize_null_scores(invalid, valid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                summarize_null_scores(valid, invalid)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            summarize_null_scores(valid, np.array([0.5, 0.5]))

    def test_invalid_cli_arguments(self) -> None:
        for flag, value in (
            ("--n-sims", "1"), ("--n-posterior-samples", "1"),
            ("--n-repeats", "1"), ("--n-bootstraps", "1"),
            ("--mira-runs", "1"), ("--seed", "-1"), ("--seed", str(2**63)),
        ):
            with self.subTest(flag=flag, value=value):
                with patch("sys.argv", ["runner", flag, value, "--dry-run"]):
                    with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                        main()
                self.assertEqual(error.exception.code, 2)

    def test_full_default_design_and_plot(self) -> None:
        seeds_seen = []

        def fake_score(truth, samples, num_runs, seed):
            self.assertEqual(tuple(truth.shape), (64, 2))
            self.assertEqual(samples.shape, (128, 64, 2))
            self.assertEqual(num_runs, 512)
            seeds_seen.append(seed)
            return float(np.random.default_rng(seed).uniform(0.6, 0.73))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = build_arg_parser().parse_args(["--output-dir", directory])
            with (
                patch("coverage_test.run_mira_null_test.mira_score", side_effect=fake_score),
                patch("coverage_test.run_mira_null_test.simulate_posteriors", wraps=simulate_posteriors) as simulate,
                patch("builtins.print"),
            ):
                run_null_test(args)
            self.assertEqual(simulate.call_count, 257)
            self.assertEqual(len(seeds_seen), 513)
            self.assertEqual(len(set(seeds_seen)), 513)
            np.testing.assert_array_equal(np.load(output / "mira_seeds.npy"), seeds_seen)
            scores = np.load(output / "scores_mira.npy")
            bootstrap = np.load(output / "bootstrap_scores_mira.npy")
            self.assertEqual(scores.shape, (256,))
            self.assertEqual(bootstrap.shape, (256,))
            systems = np.load(output / "posterior_samples.npy")
            self.assertEqual(systems.shape, (256, 128, 64, 2))
            self.assertFalse(np.array_equal(systems[0], systems[1]))
            self.assertFalse(np.array_equal(systems[0], np.load(output / "reference_posterior_samples.npy")))
            for name, shape in (
                ("ground_truth", (256, 64, 2)),
                ("data_values", (256, 64, 2)),
                ("data_cov", (256, 64, 2, 2)),
            ):
                self.assertEqual(np.load(output / f"{name}.npy").shape, shape)
            width = bootstrap.std(ddof=1)
            self.assertEqual(float(np.load(output / "bootstrap_std_mira.npy")), width)
            np.testing.assert_array_equal(np.load(output / "z_scores_mira.npy"), (scores - 2 / 3) / width)
            summary = json.loads((output / "null_summary.json").read_text())
            self.assertEqual(summary["bootstrap_std"], width)
            config = json.loads((output / "run_config.json").read_text())
            self.assertEqual(config["sigma"], 1)
            self.assertTrue(config["reference_independent_of_reruns"])
            with patch("coverage_test.plot_mira_null_test.plt.close"):
                path = plot_null_summary(output)
            fig = plt.gcf()
            self.addCleanup(plt.close, fig)
            self.assertEqual(len(fig.axes), 4)
            qq = fig.axes[1].collections[0].get_offsets()
            np.testing.assert_allclose(qq[:, 1], np.sort((scores - 2 / 3) / width))
            curve = fig.axes[0].lines[0]
            np.testing.assert_allclose(
                curve.get_ydata(), norm.pdf(curve.get_xdata(), loc=2 / 3, scale=width)
            )
            self.assertGreater(path.stat().st_size, 0)
            self.assertGreater(path.with_suffix(".png").stat().st_size, 0)

    def test_real_run_reproducibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = [Path(directory) / "first", Path(directory) / "second"]
            for output in outputs:
                args = build_arg_parser().parse_args([
                    "--output-dir", str(output), "--n-sims", "8",
                    "--n-posterior-samples", "16", "--n-repeats", "4",
                    "--n-bootstraps", "4", "--mira-runs", "4",
                ])
                with patch("builtins.print"):
                    run_null_test(args)
            for artifact in outputs[0].glob("*.npy"):
                np.testing.assert_array_equal(np.load(artifact), np.load(outputs[1] / artifact.name))
            self.assertEqual(
                (outputs[0] / "null_summary.json").read_text(),
                (outputs[1] / "null_summary.json").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
