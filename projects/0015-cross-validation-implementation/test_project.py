import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    keys = {"cv_mean_accuracy", "cv_std_accuracy", "holdout_accuracy", "n_folds", "fold_report"}
    assert keys.issubset(set(res.keys()))

    assert 0.0 <= res["cv_mean_accuracy"] <= 1.0
    assert 0.0 <= res["cv_std_accuracy"] <= 1.0
    assert 0.0 <= res["holdout_accuracy"] <= 1.0
    assert isinstance(res["n_folds"], int) and res["n_folds"] >= 2
    assert isinstance(res["fold_report"], list) and len(res["fold_report"]) == res["n_folds"]


def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    assert a == b, "run should be deterministic for the same seed"

    # Conservative floor on holdout accuracy that this synthetic pipeline should comfortably pass
    assert a["holdout_accuracy"] >= 0.70
