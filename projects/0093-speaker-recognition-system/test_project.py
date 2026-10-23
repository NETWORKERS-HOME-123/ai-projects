import json
from project import run


def test_run_returns_keys():
    r = run(42)
    assert isinstance(r, dict)
    keys = {"overall_accuracy", "known_speaker_accuracy", "unknown_reject_rate", "false_accept_rate", "threshold", "holdout_session"}
    assert keys.issubset(set(r.keys()))


def test_metrics_ranges():
    r = run(42)
    # all rates between 0 and 1
    assert 0.0 <= r["overall_accuracy"] <= 1.0
    assert 0.0 <= r["known_speaker_accuracy"] <= 1.0
    assert 0.0 <= r["unknown_reject_rate"] <= 1.0
    assert 0.0 <= r["false_accept_rate"] <= 1.0
    assert 0.0 <= r["threshold"] <= 1.0


def test_metric_floor_and_determinism():
    r1 = run(42)
    r2 = run(42)
    # Determinism on main metric
    assert r1["known_speaker_accuracy"] == r2["known_speaker_accuracy"]
    # Conservative floor: expect known speaker accuracy at least 0.60
    assert r1["known_speaker_accuracy"] >= 0.60
