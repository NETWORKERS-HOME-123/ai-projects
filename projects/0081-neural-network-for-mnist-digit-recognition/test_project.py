import json
from project import run


def test_run_returns_expected_keys():
    out = run(42)
    assert isinstance(out, dict)
    assert 'test_accuracy' in out
    assert 'per_class_accuracy' in out
    assert 'n_train' in out and 'n_test' in out


def test_metrics_in_valid_ranges():
    out = run(42)
    acc = out['test_accuracy']
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    per_class = out['per_class_accuracy']
    assert isinstance(per_class, dict)
    for v in per_class.values():
        if v is not None:
            assert 0.0 <= v <= 1.0


def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    # Determinism: exactly equal outputs for same seed
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    # Conservative floor on test accuracy
    assert a['test_accuracy'] >= 0.90
