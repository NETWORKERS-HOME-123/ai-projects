import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    keys = {"accuracy", "balanced_accuracy", "baseline_accuracy", "n_features", "top_features"}
    assert keys.issubset(out.keys())
    assert isinstance(out["accuracy"], float)
    assert isinstance(out["n_features"], int)
    assert isinstance(out["top_features"], list)


def test_metrics_ranges_and_floor():
    out = run(42)
    # valid ranges
    assert 0.0 <= out["accuracy"] <= 1.0
    assert 0.0 <= out["balanced_accuracy"] <= 1.0
    assert 0.0 <= out["baseline_accuracy"] <= 1.0
    # conservative floor for accuracy that this dataset and model comfortably pass
    assert out["accuracy"] >= 0.85


def test_determinism():
    a = run(42)
    b = run(42)
    # main metric should be exactly reproducible
    assert a["accuracy"] == b["accuracy"]
