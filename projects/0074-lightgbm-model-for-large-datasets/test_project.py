import json
from project import run


def test_run_returns_expected_keys():
    out = run(42)
    assert isinstance(out, dict)
    for key in ["sizes", "train_times_s", "memory_kb", "accuracies", "largest_accuracy"]:
        assert key in out


def test_metric_ranges_and_floor():
    out = run(42)
    sizes = out["sizes"]
    assert isinstance(sizes, list) and len(sizes) >= 1

    # train times positive and reasonable
    for t in out["train_times_s"]:
        assert isinstance(t, float) and t >= 0.0 and t < 600.0

    # memory is a non-negative integer
    for m in out["memory_kb"]:
        assert isinstance(m, int) and m >= 0

    # accuracies between 0 and 1
    for a in out["accuracies"]:
        assert 0.0 <= a <= 1.0

    # Conservative floor for the largest held-out accuracy
    largest = out["largest_accuracy"]
    assert isinstance(largest, float)
    assert largest >= 0.60


def test_determinism_of_main_metric():
    a = run(42)["largest_accuracy"]
    b = run(42)["largest_accuracy"]
    assert a == b
