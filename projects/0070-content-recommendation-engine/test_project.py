import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    expected_keys = {"recall_at_5", "topic_coverage_at_5", "new_user_recall", "baseline_recall_at_5", "sample_user", "sample_recommendations"}
    assert expected_keys.issubset(set(out.keys()))
    assert 0.0 <= out["recall_at_5"] <= 1.0
    assert 0.0 <= out["topic_coverage_at_5"] <= 1.0
    assert 0.0 <= out["new_user_recall"] <= 1.0


def test_metric_floor_and_baseline():
    out = run(42)
    assert out["recall_at_5"] >= 0.4
    assert 0.0 <= out["baseline_recall_at_5"] <= 1.0


def test_determinism():
    a = run(42)
    b = run(42)
    assert a["recall_at_5"] == b["recall_at_5"]
