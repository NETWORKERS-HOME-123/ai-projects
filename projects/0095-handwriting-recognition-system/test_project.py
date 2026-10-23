import json
from project import run


def test_run_returns_expected_keys():
    r = run(42)
    assert isinstance(r, dict)
    expected_keys = {"dataset", "vocabulary", "n_train", "n_test", "accuracy", "top3_accuracy", "per_class_accuracy_mean", "unknown_fraction", "suggestions_preview"}
    assert expected_keys.issubset(set(r.keys()))


def test_metric_ranges_and_types():
    r = run(42)
    # types
    assert isinstance(r["accuracy"], float)
    assert isinstance(r["top3_accuracy"], float)
    assert isinstance(r["unknown_fraction"], float)
    # ranges
    assert 0.0 <= r["accuracy"] <= 1.0
    assert 0.0 <= r["top3_accuracy"] <= 1.0
    assert 0.0 <= r["unknown_fraction"] <= 1.0


def test_accuracy_floor_and_determinism():
    r1 = run(42)
    r2 = run(42)
    # determinism for main metric
    assert r1["accuracy"] == r2["accuracy"]
    # conservative floor that should be comfortably met on digits with this pipeline
    assert r1["accuracy"] >= 0.9
