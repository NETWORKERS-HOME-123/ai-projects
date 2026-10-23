import json
from project import run

def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    expected_keys = {"train_size","test_size","exact_match_frac","token_accuracy","retrieval_exact_frac","examples"}
    assert expected_keys.issubset(set(out.keys()))
    assert isinstance(out["exact_match_frac"], float)
    assert isinstance(out["examples"], list)

def test_metric_ranges():
    out = run(42)
    assert 0.0 <= out["exact_match_frac"] <= 1.0
    assert 0.0 <= out["token_accuracy"] <= 1.0
    assert 0.0 <= out["retrieval_exact_frac"] <= 1.0

def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    assert a["exact_match_frac"] == b["exact_match_frac"]
    assert a["exact_match_frac"] >= 0.25
