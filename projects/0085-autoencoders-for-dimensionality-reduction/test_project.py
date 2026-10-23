import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    # Required top-level keys
    for k in ['dataset', 'model', 'latent_dims', 'results', 'ae_clf_acc_8']:
        assert k in res
    assert res['dataset'] == 'digits_builtin'
    assert isinstance(res['latent_dims'], list)
    assert isinstance(res['results'], dict)


def test_metric_ranges_and_floor():
    res = run(42)
    # Check the results entries for each latent dim
    for ld_key, entry in res['results'].items():
        assert 'latent_dim' in entry
        assert 'ae_mse' in entry and entry['ae_mse'] >= 0.0
        assert 'pca_mse' in entry and entry['pca_mse'] >= 0.0
        assert 'ae_clf_acc' in entry and 0.0 <= entry['ae_clf_acc'] <= 1.0
        assert 'pca_clf_acc' in entry and 0.0 <= entry['pca_clf_acc'] <= 1.0
    # Conservative metric floor for main reported metric ae_clf_acc_8
    assert res['ae_clf_acc_8'] >= 0.70


def test_determinism_main_metric():
    a = run(42)['ae_clf_acc_8']
    b = run(42)['ae_clf_acc_8']
    assert a == b
