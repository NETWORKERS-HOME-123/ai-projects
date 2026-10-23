import json
from project import run


def test_run_returns_expected_keys():
    metrics = run(42)
    assert isinstance(metrics, dict)
    expected = {"test_mae", "test_rmse", "test_r2", "baseline_mae", "n_train", "n_test"}
    assert expected.issubset(set(metrics.keys()))


def test_metric_ranges_and_types():
    metrics = run(42)
    # types
    assert isinstance(metrics["test_mae"], float)
    assert isinstance(metrics["test_rmse"], float)
    assert isinstance(metrics["test_r2"], float)
    # ranges
    assert metrics["test_mae"] >= 0.0
    assert metrics["test_rmse"] >= 0.0
    assert -1.0 <= metrics["test_r2"] <= 1.0
    assert metrics["n_train"] > 0 and metrics["n_test"] > 0


def test_metric_floor_and_determinism():
    m1 = run(42)
    m2 = run(42)
    # Determinism for main metric
    assert m1["test_mae"] == m2["test_mae"]
    # Conservative performance floor: model should beat a very weak threshold on MAE
    # This is a loose floor chosen to be comfortably achievable on the synthetic data.
    assert m1["test_mae"] <= 25000.0
