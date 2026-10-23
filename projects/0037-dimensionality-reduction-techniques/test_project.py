import json
from project import run


def test_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    expected_keys = {
        'pca_recon_mse', 'knn_overlap_pca', 'knn_overlap_isomap',
        'acc_original', 'acc_pca2d', 'acc_isomap2d', 'explanations'
    }
    assert expected_keys.issubset(set(out.keys()))
    # numeric types
    for k in ['pca_recon_mse', 'knn_overlap_pca', 'knn_overlap_isomap', 'acc_original', 'acc_pca2d', 'acc_isomap2d']:
        assert isinstance(out[k], float)


def test_metric_ranges_and_floor():
    out = run(42)
    # ranges
    assert 0.0 <= out['knn_overlap_pca'] <= 1.0
    assert 0.0 <= out['knn_overlap_isomap'] <= 1.0
    assert out['pca_recon_mse'] >= 0.0
    for k in ['acc_original', 'acc_pca2d', 'acc_isomap2d']:
        assert 0.0 <= out[k] <= 1.0
    # conservative metric floor for PCA 2D accuracy
    assert out['acc_pca2d'] >= 0.70


def test_determinism():
    a = run(42)
    b = run(42)
    assert a['acc_pca2d'] == b['acc_pca2d']
