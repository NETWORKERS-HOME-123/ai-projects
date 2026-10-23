import json
from project import run


def test_run_returns_expected_keys():
    res = run(42)
    assert isinstance(res, dict)
    # expected top-level keys
    expected = {
        'n_features',
        'baseline_accuracy',
        'accuracy_pca_5',
        'reconstruction_mse_5',
        'explained_variance_cumsum_5',
        'components_tried',
        'accuracy_by_k',
        'reconstruction_mse_by_k',
        'explained_variance_cumsum_by_k',
    }
    assert expected.issubset(set(res.keys()))


def test_metrics_in_valid_ranges():
    res = run(42)
    # accuracies between 0 and 1
    assert 0.0 <= res['baseline_accuracy'] <= 1.0
    assert 0.0 <= res['accuracy_pca_5'] <= 1.0
    # reconstruction mse non-negative
    assert res['reconstruction_mse_5'] >= 0.0
    # explained variance between 0 and 1
    assert 0.0 <= res['explained_variance_cumsum_5'] <= 1.0


def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    # determinism for the main metric
    assert a['accuracy_pca_5'] == b['accuracy_pca_5']
    # conservative metric floor that should comfortably pass on this dataset
    assert a['accuracy_pca_5'] >= 0.85
