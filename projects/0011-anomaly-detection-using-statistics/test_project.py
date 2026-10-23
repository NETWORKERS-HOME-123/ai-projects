import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    # required keys
    for k in ("precision", "recall", "tp", "fp", "fn", "held_out_size"):
        assert k in res
    assert 0.0 <= res["precision"] <= 1.0
    assert 0.0 <= res["recall"] <= 1.0


def test_metric_floor_and_ranges():
    res = run(42)
    # Conservative floor: precision should be at least 0.60 on this synthetic setup
    assert res["precision"] >= 0.60
    # counts are non-negative integers
    assert isinstance(res["tp"], int) and res["tp"] >= 0
    assert isinstance(res["fp"], int) and res["fp"] >= 0
    assert isinstance(res["fn"], int) and res["fn"] >= 0


def test_determinism():
    a = run(42)
    b = run(42)
    # Compare main metric precisely
    assert a["precision"] == b["precision"]
    assert a == b
