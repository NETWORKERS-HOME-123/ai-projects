import json
from project import run


def test_run_keys_and_types():
    r = run(42)
    assert isinstance(r, dict)
    assert "categorical_schema" in r
    assert "onehot_baseline" in r and "ordinal_gb" in r
    assert "accuracy_test" in r["ordinal_gb"]
    assert "roc_auc_test" in r["ordinal_gb"]


def test_metric_ranges_and_floor():
    r = run(42)
    acc = r["ordinal_gb"]["accuracy_test"]
    roc = r["ordinal_gb"]["roc_auc_test"]
    # sensible ranges
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= roc <= 1.0
    # conservative floor: model should be reasonably predictive on synthetic task
    assert acc >= 0.70


def test_determinism_main_metric():
    a = run(42)["ordinal_gb"]["accuracy_test"]
    b = run(42)["ordinal_gb"]["accuracy_test"]
    assert a == b
