import json

from project import run


def test_run_keys_and_types():
    r = run(42)
    # required top-level keys
    keys = [
        "accuracy_tree",
        "accuracy_rf",
        "accuracy_voting",
        "per_class",
        "error_overlap",
        "inference_cost_proxy",
        "n_test_samples",
        "seed",
    ]
    for k in keys:
        assert k in r
    # types
    assert isinstance(r["accuracy_rf"], float)
    assert isinstance(r["n_test_samples"], int)
    assert isinstance(r["per_class"], dict)


def test_metrics_ranges_and_floor():
    r = run(42)
    # values in [0,1]
    for a in ["accuracy_tree", "accuracy_rf", "accuracy_voting"]:
        assert 0.0 <= r[a] <= 1.0
    # conservative floor for Random Forest on Iris
    assert r["accuracy_rf"] >= 0.85


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    # determinism for main metric
    assert a["accuracy_rf"] == b["accuracy_rf"]
