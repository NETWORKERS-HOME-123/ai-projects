import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    # Expected keys
    expect = [
        "accuracy_standard",
        "accuracy_minmax",
        "accuracy_robust",
        "best_scaler",
        "outlier_test_indices",
        "outlier_examples",
        "saved_transform",
        "n_train",
        "n_test",
    ]
    for k in expect:
        assert k in res

    assert isinstance(res["accuracy_standard"], float)
    assert isinstance(res["best_scaler"], str)
    assert isinstance(res["n_train"], int)


def test_metric_ranges_and_floor():
    res = run(42)
    # Basic sanity ranges
    for key in ["accuracy_standard", "accuracy_minmax", "accuracy_robust"]:
        v = res[key]
        assert 0.0 <= v <= 1.0

    # Conservative floor on main metric
    assert res["accuracy_standard"] >= 0.70


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    # Compare main metric and entire dict for determinism
    assert a == b
