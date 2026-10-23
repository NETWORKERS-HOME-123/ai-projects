import json

import pytest
from project import run


EXPECTED_KEYS = {
    "n_samples",
    "n_features",
    "feature_names",
    "auc_test",
    "brier_test",
    "accuracy_test",
    "auc_baseline",
    "time_seconds_boosted",
    "packages",
}


def test_run_keys():
    res = run(42)
    assert isinstance(res, dict)
    assert EXPECTED_KEYS.issubset(set(res.keys()))


def test_metrics_ranges_and_floor():
    res = run(42)
    auc = res["auc_test"]
    brier = res["brier_test"]
    acc = res["accuracy_test"]

    # basic ranges
    assert 0.0 <= auc <= 1.0
    assert 0.0 <= brier <= 1.0
    assert 0.0 <= acc <= 1.0

    # conservative floor for held-out ROC AUC (should comfortably pass on this synthetic problem)
    assert auc >= 0.75


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    # determinism on the main metric
    assert a["auc_test"] == b["auc_test"]
