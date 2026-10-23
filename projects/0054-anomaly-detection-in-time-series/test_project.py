from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    keys = [
        "event_recall",
        "false_alerts_per_day",
        "mean_detection_delay_hours",
        "alerts_count",
        "annotated_events_considered",
        "threshold_used",
        "review_timeline_preview",
    ]
    for k in keys:
        assert k in res
    assert 0.0 <= res["event_recall"] <= 1.0
    assert res["false_alerts_per_day"] >= 0.0
    assert isinstance(res["alerts_count"], int)
    assert isinstance(res["annotated_events_considered"], int)


def test_metric_floor_and_ranges():
    res = run(42)
    assert res["annotated_events_considered"] >= 1
    assert res["event_recall"] >= 0.60
    assert res["false_alerts_per_day"] <= 10.0


def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
