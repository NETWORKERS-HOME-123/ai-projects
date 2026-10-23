import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    keys = {'online_mean_accuracy', 'frozen_mean_accuracy', 'drift_fraction', 'resets', 'n_batches'}
    assert keys.issubset(set(res.keys()))
    assert 0.0 <= res['online_mean_accuracy'] <= 1.0
    assert 0.0 <= res['frozen_mean_accuracy'] <= 1.0
    assert 0.0 <= res['drift_fraction'] <= 1.0
    assert isinstance(res['resets'], int)
    assert isinstance(res['n_batches'], int)


def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    # determinism
    assert a == b
    # conservative floor for the main held-out metric: online_mean_accuracy
    assert a['online_mean_accuracy'] >= 0.70
