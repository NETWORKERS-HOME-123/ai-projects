import json
from project import run


def test_run_returns_expected_keys():
    metrics = run(42)
    expected = {"auc", "brier", "threshold", "fpr_region_0", "fpr_region_1", "model_card"}
    assert set(metrics.keys()) == expected


def test_metric_ranges_and_floor():
    metrics = run(42)
    auc = metrics["auc"]
    brier = metrics["brier"]
    f0 = metrics["fpr_region_0"]
    f1 = metrics["fpr_region_1"]

    # Ranges
    assert 0.0 <= auc <= 1.0
    assert 0.0 <= brier <= 1.0
    assert 0.0 <= f0 <= 1.0
    assert 0.0 <= f1 <= 1.0

    # Conservative performance floor on AUC
    assert auc >= 0.75


def test_determinism_of_main_metric():
    m1 = run(42)["auc"]
    m2 = run(42)["auc"]
    assert m1 == m2
