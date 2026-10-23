import json
from project import run


def test_run_keys():
    res = run(42)
    assert isinstance(res, dict)
    # expected keys
    for k in ('accuracy', 'baseline_accuracy', 'n_classes', 'per_class_error_mean'):
        assert k in res


def test_metric_ranges():
    res = run(42)
    assert 0.0 <= res['accuracy'] <= 1.0
    assert 0.0 <= res['baseline_accuracy'] <= 1.0
    assert isinstance(res['n_classes'], int) and res['n_classes'] >= 2
    assert 0.0 <= res['per_class_error_mean'] <= 1.0


def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    # determinism
    assert a == b
    # conservative floor for digits on this pipeline
    assert a['accuracy'] >= 0.85
