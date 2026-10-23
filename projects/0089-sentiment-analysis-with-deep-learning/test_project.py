import json

import pytest
from project import run


EXPECTED_KEYS = {
    "macro_f1_neural",
    "macro_f1_tfidf",
    "brier_neural",
    "brier_tfidf",
    "infer_time_ratio_mlp_over_baseline",
    "train_time_neural",
    "train_time_tfidf",
    "review_samples",
    "n_test",
}


def test_run_returns_expected_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    assert EXPECTED_KEYS.issubset(set(res.keys()))
    assert isinstance(res["macro_f1_neural"], float)
    assert isinstance(res["review_samples"], list)
    assert isinstance(res["n_test"], int)


def test_metrics_in_ranges_and_floor():
    res = run(42)
    # metric ranges
    assert 0.0 <= res["macro_f1_neural"] <= 1.0
    assert 0.0 <= res["macro_f1_tfidf"] <= 1.0
    assert 0.0 <= res["brier_neural"] <= 1.0
    assert 0.0 <= res["brier_tfidf"] <= 1.0
    # conservative floor for the neural macro F1
    assert res["macro_f1_neural"] >= 0.60


def test_determinism_of_main_metric():
    a = run(42)["macro_f1_neural"]
    b = run(42)["macro_f1_neural"]
    assert a == b
