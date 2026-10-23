import json
from project import run


def test_run_returns_expected_keys():
    res = run(42)
    assert isinstance(res, dict)
    expected_keys = {
        "r2",
        "mae_model",
        "mae_baseline_signup_group",
        "group_mae_mean",
        "band_mae_low",
        "band_mae_mid",
        "band_mae_high",
        "planning_total_pred",
        "planning_total_uncertainty_se",
        "n_test",
        "model",
    }
    assert expected_keys.issubset(set(res.keys()))


def test_metrics_in_valid_ranges():
    res = run(42)
    assert -1.0 <= res["r2"] <= 1.0
    assert res["mae_model"] >= 0.0
    assert res["mae_baseline_signup_group"] >= 0.0
    assert isinstance(res["n_test"], int) and res["n_test"] > 0


def test_metric_floor_and_determinism():
    r = run(42)
    assert r["r2"] >= 0.30
    r2_a = run(42)["r2"]
    r2_b = run(42)["r2"]
    assert r2_a == r2_b
