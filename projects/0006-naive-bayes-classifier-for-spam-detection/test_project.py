import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    expected_keys = {
        "accuracy",
        "spam_recall",
        "ham_false_positive_rate",
        "majority_accuracy",
        "top_tokens",
        "n_train",
        "n_test",
    }
    assert set(out.keys()) == expected_keys
    assert isinstance(out["top_tokens"], list)


def test_metric_ranges_and_floor():
    out = run(42)
    # basic ranges
    assert 0.0 <= out["accuracy"] <= 1.0
    assert 0.0 <= out["spam_recall"] <= 1.0
    assert 0.0 <= out["ham_false_positive_rate"] <= 1.0
    # conservative floor for spam recall (main held-out metric)
    assert out["spam_recall"] >= 0.75


def test_determinism():
    a = run(42)
    b = run(42)
    # determinism on the main metric
    assert a["spam_recall"] == b["spam_recall"]
