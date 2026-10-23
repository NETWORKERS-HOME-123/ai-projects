import json
from project import run
import math


def test_keys_present_and_types():
    out = run(42)
    assert isinstance(out, dict)
    expected = {'auc', 'brier', 'precision_at_capacity', 'precision_recency', 'subgroup_coverage_ratio', 'min_service_rate'}
    assert expected.issubset(set(out.keys()))
    for k in expected:
        assert isinstance(out[k], float)


def test_ranges_and_floor():
    out = run(42)
    # basic ranges
    assert 0.0 <= out['auc'] <= 1.0
    assert 0.0 <= out['brier'] <= 1.0
    assert 0.0 <= out['precision_at_capacity'] <= 1.0
    assert 0.0 <= out['precision_recency'] <= 1.0
    assert 0.0 <= out['subgroup_coverage_ratio'] or math.isfinite(out['subgroup_coverage_ratio'])
    assert 0.0 <= out['min_service_rate'] <= 1.0
    # conservative floor: model should reach reasonable precision at capacity on this synthetic dataset
    assert out['precision_at_capacity'] >= 0.50


def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
