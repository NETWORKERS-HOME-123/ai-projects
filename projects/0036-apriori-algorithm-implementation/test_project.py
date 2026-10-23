import json
from project import run

def test_run_keys_and_types():
    r = run(42)
    assert isinstance(r, dict)
    keys = {"agreement", "num_transactions", "min_support_used", "num_frequent_itemsets", "max_candidate_count", "iterations"}
    assert keys.issubset(set(r.keys()))
    assert 0.0 <= float(r["agreement"]) <= 1.0
    assert isinstance(r["num_transactions"], int) and r["num_transactions"] > 0


def test_metric_floor_and_ranges():
    r = run(42)
    # conservative floor: exact agreement with exhaustive enumeration expected on this tiny dataset
    assert r["agreement"] >= 1.0
    assert 0.0 <= r["min_support_used"] <= 1.0
    assert r["num_frequent_itemsets"] >= 0
    assert r["max_candidate_count"] >= 0


def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
