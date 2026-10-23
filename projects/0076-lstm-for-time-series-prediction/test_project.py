import json
from project import run
import pytest


def test_run_returns_expected_keys():
    r = run(42)
    assert isinstance(r, dict)
    keys = [
        'seed', 'window_size', 'horizon_test_points',
        'test_r2_mlp', 'test_mae_mlp',
        'test_r2_persistence', 'test_mae_persistence',
        'test_r2_ridge', 'test_mae_ridge',
        'example_window', 'example_prediction_mlp'
    ]
    for k in keys:
        assert k in r


def test_metrics_ranges_and_floor():
    r = run(42)
    # R2 should be within -1..1
    assert -1.0 <= r['test_r2_mlp'] <= 1.0
    assert -1.0 <= r['test_r2_persistence'] <= 1.0
    assert -1.0 <= r['test_r2_ridge'] <= 1.0
    # MAE should be non-negative
    assert r['test_mae_mlp'] >= 0.0
    assert r['test_mae_persistence'] >= 0.0
    assert r['test_mae_ridge'] >= 0.0
    # Conservative floor on main held-out metric (R2) that should be comfortably met
    assert r['test_r2_mlp'] >= 0.15


def test_determinism_of_main_metric():
    a = run(42)
    b = run(42)
    assert a['test_r2_mlp'] == b['test_r2_mlp']
