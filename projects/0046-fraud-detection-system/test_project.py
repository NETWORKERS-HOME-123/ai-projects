from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    keys = {'precision_at_capacity', 'recall_at_capacity', 'isolation_precision', 'rule_precision', 'review_capacity', 'n_verified_in_queue', 'n_missed_verified', 'dataset_size'}
    assert keys.issubset(set(res.keys()))
    assert isinstance(res['precision_at_capacity'], float)
    assert isinstance(res['review_capacity'], int)


def test_metric_ranges_and_floor():
    res = run(42)
    assert 0.0 <= res['precision_at_capacity'] <= 1.0
    assert 0.0 <= res['recall_at_capacity'] <= 1.0
    assert 0.0 <= res['isolation_precision'] <= 1.0
    assert 0.0 <= res['rule_precision'] <= 1.0
    assert res['precision_at_capacity'] >= 0.5


def test_determinism_main_metric():
    a = run(42)['precision_at_capacity']
    b = run(42)['precision_at_capacity']
    assert a == b
