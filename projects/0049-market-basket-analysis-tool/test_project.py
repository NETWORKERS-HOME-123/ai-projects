import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    keys = {'n_transactions','n_items','n_rules','avg_train_lift','promo_support','nonpromo_support','held_out_agreement'}
    assert keys.issubset(set(res.keys()))
    assert isinstance(res['n_transactions'], int)
    assert isinstance(res['n_items'], int)
    assert isinstance(res['n_rules'], int)
    for k in ('avg_train_lift','promo_support','nonpromo_support','held_out_agreement'):
        assert isinstance(res[k], float)


def test_ranges_and_sanity():
    res = run(42)
    assert res['n_transactions'] > 0
    assert 0.0 <= res['promo_support'] <= 1.0
    assert 0.0 <= res['nonpromo_support'] <= 1.0
    assert 0.0 <= res['avg_train_lift'] or res['avg_train_lift'] >= -10.0
    assert 0.0 <= res['held_out_agreement'] <= 1.0


def test_metric_floor():
    res = run(42)
    # Conservative floor: at least 40% of top rules appear in held-out data
    assert res['held_out_agreement'] >= 0.4


def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
