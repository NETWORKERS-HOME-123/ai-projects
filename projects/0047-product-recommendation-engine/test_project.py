from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    expected = {"recall_at_5", "coverage", "popularity_recall_at_5", "new_user_recall"}
    assert set(out.keys()) == expected
    for v in out.values():
        assert isinstance(v, float) or isinstance(v, int)


def test_metric_ranges_and_floor():
    out = run(42)
    assert 0.0 <= out["recall_at_5"] <= 1.0
    assert 0.0 <= out["coverage"] <= 1.0
    assert 0.0 <= out["popularity_recall_at_5"] <= 1.0
    assert 0.0 <= out["new_user_recall"] <= 1.0
    assert out["recall_at_5"] >= 0.35


def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
