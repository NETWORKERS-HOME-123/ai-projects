import json
from project import run

def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    keys = {
        "selectk_mean_cv",
        "rfe_mean_cv",
        "selectk_test_accuracy",
        "rfe_test_accuracy",
        "selectk_feature_count",
        "rfe_feature_count",
        "selectk_stability",
        "rfe_stability",
        "audit",
    }
    assert keys.issubset(set(out.keys()))
    assert isinstance(out["selectk_test_accuracy"], float)
    assert isinstance(out["selectk_feature_count"], int)
    assert isinstance(out["audit"], dict)

def test_metric_ranges_and_floor():
    out = run(42)
    assert 0.0 <= out["selectk_test_accuracy"] <= 1.0
    assert 0.0 <= out["rfe_test_accuracy"] <= 1.0
    assert 0.0 <= out["selectk_stability"] <= 1.0
    assert 0.0 <= out["rfe_stability"] <= 1.0
    assert out["selectk_test_accuracy"] >= 0.75

def test_determinism_of_main_metric():
    a = run(42)
    b = run(42)
    assert a["selectk_test_accuracy"] == b["selectk_test_accuracy"]
