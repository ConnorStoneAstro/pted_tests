from pathlib import Path
import sys
import tempfile

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.datasets.vision import generate_gaussian_image_problem
from benchmarks.plots import plot_gaussian_image_problem, plot_method_sweep
from benchmarks.datasets.gaussian import generate_gaussian_problem
from benchmarks.datasets.vision import class_split_problem_layout, pair_blend_problem_layout
from benchmarks.metrics import (
    fid_score,
    fld_score,
    ks_pc1_pvalue,
)
from benchmarks.runners.gaussian import run_gaussian_sweep
from benchmarks.runners.vision import run_gaussian_image_sweep, run_vision_pair_blend_sweep


def main() -> None:
    problem = generate_gaussian_problem(severity=0.0, deviation="mean_shift", n_samples=32, seed=7)
    assert problem.x.shape == (32, 1)
    assert problem.y.shape == (32, 1)

    null_problem = generate_gaussian_problem(
        severity=0.0,
        deviation="mean_shift",
        n_samples=64,
        seed=11,
    )
    shifted_problem = generate_gaussian_problem(
        severity=2.0,
        deviation="mean_shift",
        n_samples=64,
        seed=11,
    )

    null_ks = ks_pc1_pvalue(null_problem.x, null_problem.y)
    shifted_ks = ks_pc1_pvalue(shifted_problem.x, shifted_problem.y)
    assert 0.0 <= null_ks <= 1.0
    assert 0.0 <= shifted_ks <= 1.0
    assert shifted_ks <= null_ks or np.isclose(shifted_ks, null_ks)

    null_fld = fld_score(null_problem.x, null_problem.y)
    shifted_fld = fld_score(shifted_problem.x, shifted_problem.y)
    assert np.isfinite(null_fld)
    assert np.isfinite(shifted_fld)
    assert shifted_fld >= null_fld or np.isclose(shifted_fld, null_fld)

    null_fid = fid_score(null_problem.x, null_problem.y)
    shifted_fid = fid_score(shifted_problem.x, shifted_problem.y)
    assert shifted_fid >= null_fid or np.isclose(shifted_fid, null_fid)

    raw_fld = fld_score(problem.x, problem.y)
    raw_fid = fid_score(problem.x, problem.y)
    assert np.isfinite(raw_fld)
    assert raw_fid >= 0.0

    records = run_gaussian_sweep(
        deviations=("mean_shift",),
        severities=(0.0, 1.0),
        n_samples=16,
        permutations=10,
        seeds=(0,),
    )
    required_methods = {"pted", "ks_pc1", "fld", "fid", "pqm"}
    assert {record["method"] for record in records} == required_methods
    expected_rows = 2 * 1 * len(required_methods)
    assert len(records) == expected_rows
    assert all("score" in record for record in records)
    assert all("raw_score" in record for record in records)

    pqm_rows = [record for record in records if record["method"] == "pqm"]
    assert len(pqm_rows) > 0
    assert np.all(np.isfinite([float(record["score"]) for record in pqm_rows]))
    assert np.all([(0.0 <= float(record["score"]) <= 1.0) for record in pqm_rows])

    mnist_plan = class_split_problem_layout(
        dataset="mnist", class_a=0, class_b=1, severities=(0.0, 0.5, 1.0)
    )
    cifar_plan = class_split_problem_layout(
        dataset="cifar10", class_a=2, class_b=3, severities=(0.0, 0.5, 1.0)
    )
    assert len(mnist_plan) == 3
    assert len(cifar_plan) == 3

    image_problem = generate_gaussian_image_problem(severity=0.5, n_samples=32, seed=3)
    assert image_problem.x.shape == (32, 2, 2)
    assert image_problem.y.shape == (32, 2, 2)

    image_records = run_gaussian_image_sweep(
        severities=(0.0, 0.5, 1.0),
        n_samples=32,
        permutations=10,
        seeds=(0,),
    )
    vision_methods = required_methods
    assert {record["method"] for record in image_records} == vision_methods
    expected_image_rows = 3 * 1 * len(vision_methods)
    assert len(image_records) == expected_image_rows
    assert all("score" in record for record in image_records)
    assert all("raw_score" in record for record in image_records)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        plot_method_sweep(
            records,
            tmpdir_path / "gaussian_sweep.png",
            title="Gaussian sweep",
            normalize_methods={"fld": "inverse", "fid": "inverse"},
        )
        plot_method_sweep(
            image_records,
            tmpdir_path / "vision_sweep.png",
            title="2x2 vision sweep",
            normalize_methods={"fld": "inverse", "fid": "inverse"},
        )
        plot_gaussian_image_problem(
            image_problem, tmpdir_path / "vision_problem.png", title="2x2 Gaussian image"
        )
        assert (tmpdir_path / "gaussian_sweep.png").exists()
        assert (tmpdir_path / "vision_sweep.png").exists()
        assert (tmpdir_path / "vision_problem.png").exists()

    blend_problems = pair_blend_problem_layout(
        dataset="mnist", severities=(0.0, 0.25, 0.5), n_samples=32, seed=2
    )

    def _fake_pool_loader(_problem):
        rng = np.random.default_rng(123)
        return rng.normal(size=(128, 2, 2)).astype(np.float32)

    blend_records = run_vision_pair_blend_sweep(
        blend_problems,
        loader=_fake_pool_loader,
        permutations=10,
    )
    assert len(blend_records) == 3 * len(vision_methods)
    assert {record["method"] for record in blend_records} == {
        "pted",
        "ks_pc1",
        "fld",
        "fid",
        "pqm",
    }


if __name__ == "__main__":
    main()
