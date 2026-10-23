import json
from project import run


def test_run_returns_keys():
    out = run(42)
    assert isinstance(out, dict)
    expected_keys = {"accuracy", "best_k", "cv_best_score", "hand_knn_accuracy", "misleading_nearest_count", "n_test"}
    assert expected_keys.issubset(set(out.keys()))


def test_metric_ranges_and_floor():
    out = run(42)
    # accuracy in [0,1]
    acc = out["accuracy"]
    assert 0.0 <= acc <= 1.0
    # conservative floor: accuracy should be at least 0.70 for this synthetic task
    assert acc >= 0.70
    # hand-built accuracy also reasonable
    hand = out["hand_knn_accuracy"]
    assert 0.0 <= hand <= 1.0
    # counts are integers and plausible
    assert isinstance(out["misleading_nearest_count"], int)
    assert 0 <= out["misleading_nearest_count"] <= out["n_test"]


def test_determinism_of_main_metric():
    a = run(42)["accuracy"]
    b = run(42)["accuracy"]
    assert a == b
