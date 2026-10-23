import json
from project import run


def test_run_returns_keys():
    out = run(42)
    assert isinstance(out, dict)
    expected = {'top1_accuracy', 'false_match_rate', 'false_non_match_rate', 'unknown_reject_rate', 'threshold', 'gallery_size'}
    assert expected.issubset(set(out.keys()))


def test_metric_ranges():
    out = run(42)
    # All rates are between 0 and 1
    for k in ['top1_accuracy', 'false_match_rate', 'false_non_match_rate', 'unknown_reject_rate']:
        assert 0.0 <= out[k] <= 1.0
    assert out['gallery_size'] >= 2
    assert 0.0 <= out['threshold'] <= 1.0


def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    # Deterministic for the same seed (main metric equality)
    assert a['top1_accuracy'] == b['top1_accuracy']
    # Conservative floor for top-1 accuracy (educational expectation)
    assert a['top1_accuracy'] >= 0.70
