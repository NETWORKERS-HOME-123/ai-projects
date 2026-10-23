import json
from project import run


def test_run_returns_expected_keys():
    res = run(42)
    assert isinstance(res, dict)
    for k in ["best_params", "best_cv_score", "test_accuracy", "experiment_table"]:
        assert k in res


def test_numeric_ranges_and_floor():
    res = run(42)
    best_cv = res["best_cv_score"]
    test_acc = res["test_accuracy"]
    # basic ranges
    assert 0.0 <= best_cv <= 1.0
    assert 0.0 <= test_acc <= 1.0
    # conservative floor for held-out accuracy
    assert test_acc >= 0.90


def test_experiments_table_and_determinism():
    a = run(42)
    b = run(42)
    # determinism for main metric
    assert a["best_cv_score"] == b["best_cv_score"]
    assert a["test_accuracy"] == b["test_accuracy"]
    # experiment table is non-empty and well formed
    table = a["experiment_table"]
    assert isinstance(table, list)
    assert len(table) >= 1
    first = table[0]
    assert "params" in first and "mean_cv_score" in first and "mean_fit_time_sec" in first
