import json
from project import run

def test_run_has_expected_keys():
    r = run(42)
    assert isinstance(r, dict)
    expected = {
        "precision_at_3",
        "baseline_precision_at_3",
        "coverage",
        "baseline_coverage",
        "example_recommendation_for_u1",
        "cold_start_recommendation",
    }
    assert expected.issubset(set(r.keys()))

def test_metrics_in_ranges():
    r = run(42)
    assert 0.0 <= float(r["precision_at_3"]) <= 1.0
    assert 0.0 <= float(r["baseline_precision_at_3"]) <= 1.0
    assert 0.0 <= float(r["coverage"]) <= 1.0
    assert 0.0 <= float(r["baseline_coverage"]) <= 1.0

def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    assert a["precision_at_3"] == b["precision_at_3"]
    assert float(a["precision_at_3"]) >= 0.2
