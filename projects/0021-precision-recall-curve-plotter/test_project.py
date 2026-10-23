import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    keys = {"average_precision", "pr_auc_trap", "prevalence", "thresholds_info", "review_queue", "seed"}
    assert keys.issubset(set(out.keys()))
    assert 0.0 <= out["average_precision"] <= 1.0
    assert 0.0 <= out["pr_auc_trap"] <= 1.0
    assert 0.0 <= out["prevalence"] <= 1.0


def test_metric_floor_and_ranges():
    out = run(42)
    # Conservative floor for average precision on this synthetic task
    assert out["average_precision"] >= 0.15
    # lengths for PR arrays should be reasonable
    pr_points = out["precision_recall_points"]
    assert pr_points["precision_length"] >= 2
    assert pr_points["recall_length"] >= 2


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    # main metric must be identical for same seed
    assert a["average_precision"] == b["average_precision"]
