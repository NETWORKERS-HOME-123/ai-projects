import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    # required keys
    for k in ("average_precision", "brier_score", "recency_rule_ap", "precision_at_recall_0.5", "review_queue_sample", "n_test"):
        assert k in out
    assert isinstance(out["average_precision"], float)
    assert isinstance(out["brier_score"], float)
    assert isinstance(out["n_test"], int)
    assert isinstance(out["review_queue_sample"], list)


def test_metric_ranges_and_floor():
    out = run(42)
    ap = out["average_precision"]
    brier = out["brier_score"]
    ap_rec = out["recency_rule_ap"]
    # Ranges
    assert 0.0 <= ap <= 1.0
    assert 0.0 <= ap_rec <= 1.0
    # Brier is between 0 and 1 for binary
    assert 0.0 <= brier <= 1.0
    # Conservative floor for average precision (held-out)
    assert ap >= 0.45


def test_determinism_main_metric():
    a = run(42)["average_precision"]
    b = run(42)["average_precision"]
    assert a == b
