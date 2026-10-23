import json
from project import run


def test_run_keys_and_types():
    r = run(42)
    assert isinstance(r, dict)
    expected_keys = {"dataset", "n_samples", "n_features", "n_classes", "cv_folds", "models", "best_model", "best_valid_mean", "generalisation_gap", "recommendation"}
    assert expected_keys.issubset(set(r.keys()))

    # models structure
    models = r["models"]
    for m in ("logistic", "random_forest"):
        assert m in models
        for key in ("train_sizes_abs", "train_score_mean", "train_score_std", "valid_score_mean", "valid_score_std"):
            assert key in models[m]


def test_metric_ranges_and_floor():
    r = run(42)
    best = r["best_valid_mean"]
    # Ranges
    assert 0.0 <= best <= 1.0
    # Conservative floor that should be comfortably met on Wine dataset
    assert best >= 0.70

    # Check arrays lengths and ranges
    for m in r["models"]:
        vals = r["models"][m]["valid_score_mean"]
        assert all(0.0 <= v <= 1.0 for v in vals)
        assert len(vals) >= 3


def test_determinism():
    a = run(42)["best_valid_mean"]
    b = run(42)["best_valid_mean"]
    assert a == b
