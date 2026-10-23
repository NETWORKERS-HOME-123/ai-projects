import json
from project import run

def test_run_keys():
    res = run(42)
    assert isinstance(res, dict)
    expected_keys = {
        'n_documents', 'n_train', 'n_hold', 'k', 'silhouette', 'stability_agreement',
        'cluster_sizes', 'top_terms', 'baseline_group_sizes', 'editable_labels', 'example_move_action', 'seed'
    }
    assert expected_keys.issubset(set(res.keys()))

def test_metric_ranges_and_floor():
    res = run(42)
    assert res['n_documents'] >= 8
    assert res['n_train'] >= 1
    assert res['n_hold'] >= 1
    sil = res['silhouette']
    if sil is not None:
        assert -1.0 <= sil <= 1.0
        assert sil >= 0.05
    stab = res['stability_agreement']
    assert 0.0 <= stab <= 1.0

def test_determinism():
    a = run(42)
    b = run(42)
    assert a['seed'] == b['seed']
    assert a['silhouette'] == b['silhouette']
    assert a['stability_agreement'] == b['stability_agreement']
