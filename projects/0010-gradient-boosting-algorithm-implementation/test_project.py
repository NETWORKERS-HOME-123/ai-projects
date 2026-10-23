import json
from project import run


def test_run_returns_expected_keys():
    res = run(42)
    assert isinstance(res, dict)
    expected_keys = {
        "baseline_mostfreq_accuracy",
        "logreg_accuracy",
        "gbc_best_params",
        "gbc_accuracy",
        "gbc_precision",
        "gbc_recall",
        "gbc_brier_score",
        "invite_fraction",
        "invite_precision",
    }
    assert expected_keys.issubset(set(res.keys()))


def test_metrics_valid_ranges():
    res = run(42)
    # Accuracies and precision/recall in [0,1]
    assert 0.0 <= res["baseline_mostfreq_accuracy"] <= 1.0
    assert 0.0 <= res["logreg_accuracy"] <= 1.0
    assert 0.0 <= res["gbc_accuracy"] <= 1.0
    assert 0.0 <= res["gbc_precision"] <= 1.0
    assert 0.0 <= res["gbc_recall"] <= 1.0
    assert 0.0 <= res["invite_fraction"] <= 1.0
    assert 0.0 <= res["invite_precision"] <= 1.0
    # Brier score is non-negative
    assert res["gbc_brier_score"] >= 0.0


def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    # Determinism on the main metric (gbc_accuracy)
    assert a["gbc_accuracy"] == b["gbc_accuracy"]
    # Conservative floor: expect accuracy >= 0.70 on this synthetic example
    assert a["gbc_accuracy"] >= 0.70
