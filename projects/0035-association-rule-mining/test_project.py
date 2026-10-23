import json
from project import run


def test_run_keys_and_types():
    r = run(42)
    assert isinstance(r, dict)
    keys = {"n_transactions_train", "n_transactions_test", "n_rules", "avg_lift", "stability_fraction", "top_rules"}
    assert keys.issubset(set(r.keys()))
    assert isinstance(r["n_transactions_train"], int)
    assert isinstance(r["n_transactions_test"], int)
    assert isinstance(r["n_rules"], int)
    assert isinstance(r["avg_lift"], float)
    assert isinstance(r["stability_fraction"], float)
    assert isinstance(r["top_rules"], list)


def test_metric_ranges_and_floor():
    r = run(42)
    assert r["n_transactions_train"] > 0
    assert r["n_transactions_test"] > 0
    assert r["n_rules"] >= 0
    assert 0.0 <= r["avg_lift"] <= 1e6
    assert 0.0 <= r["stability_fraction"] <= 1.0
    # conservative floor: at least 30% of rules are stable on held-out (this dataset is designed to pass)
    assert r["stability_fraction"] >= 0.30


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    # main metric is stability_fraction
    assert a["stability_fraction"] == b["stability_fraction"]
