import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    expected_keys = {"roc_auc", "average_precision", "roc_auc_reversed_labels", "roc_auc_tied_scores", "positive_label", "threshold_table"}
    assert expected_keys.issubset(set(out.keys()))
    assert isinstance(out["roc_auc"], float)
    assert isinstance(out["average_precision"], float)
    assert isinstance(out["positive_label"], int)
    assert isinstance(out["threshold_table"], list)


def test_metric_ranges_and_floor():
    out = run(42)
    roc_auc = out["roc_auc"]
    ap = out["average_precision"]
    # Basic valid ranges
    assert 0.0 <= roc_auc <= 1.0
    assert 0.0 <= ap <= 1.0
    # Conservative floor: expect a reasonable ranking ability on this synthetic task
    assert roc_auc >= 0.70


def test_determinism_of_main_metric():
    a = run(42)
    b = run(42)
    # Determinism for main metric
    assert a["roc_auc"] == b["roc_auc"]
