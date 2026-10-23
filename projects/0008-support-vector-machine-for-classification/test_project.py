import json
import numpy as np
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    expected_keys = {
        'kernel_chosen', 'accuracy', 'precision', 'recall', 'f1',
        'false_negatives', 'false_negative_rate', 'uncertain_count',
        'train_cv_linear_mean', 'train_cv_rbf_mean'
    }
    assert expected_keys.issubset(set(res.keys()))
    # types
    assert isinstance(res['kernel_chosen'], str)
    for k in ['accuracy', 'precision', 'recall', 'f1', 'false_negative_rate', 'train_cv_linear_mean', 'train_cv_rbf_mean']:
        assert isinstance(res[k], float)
    for k in ['false_negatives', 'uncertain_count']:
        assert isinstance(res[k], int)


def test_metric_ranges_and_floor():
    res = run(42)
    # basic ranges
    assert 0.0 <= res['accuracy'] <= 1.0
    assert 0.0 <= res['precision'] <= 1.0
    assert 0.0 <= res['recall'] <= 1.0
    assert 0.0 <= res['f1'] <= 1.0
    assert 0.0 <= res['false_negative_rate'] <= 1.0
    # conservative accuracy floor that should be comfortably met on this synthetic task
    assert res['accuracy'] >= 0.78


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    # main metric determinism: accuracy and kernel chosen must match
    assert a['accuracy'] == b['accuracy']
    assert a['kernel_chosen'] == b['kernel_chosen']
