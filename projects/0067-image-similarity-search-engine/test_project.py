import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    expected_keys = {"recall_at_1", "recall_at_5", "mrr", "catalogue_size", "query_size", "method", "seed", "ranked_provenance_sample"}
    assert expected_keys.issubset(set(res.keys()))


def test_ranges_and_sizes():
    res = run(42)
    assert 0.0 <= res["recall_at_1"] <= 1.0
    assert 0.0 <= res["recall_at_5"] <= 1.0
    assert 0.0 <= res["mrr"] <= 1.0
    assert isinstance(res["catalogue_size"], int) and res["catalogue_size"] > 0
    assert isinstance(res["query_size"], int) and res["query_size"] > 0


def test_recall_floor_and_determinism():
    r1 = run(42)
    r2 = run(42)
    assert r1 == r2
    # conservative floor on recall@5
    assert r1["recall_at_5"] >= 0.7
