import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    # required keys
    for k in ['version', 'accuracy', 'roc_auc', 'accuracy_baseline', 'n_train', 'n_test', 'example_input_rows']:
        assert k in out
    assert isinstance(out['version'], str)
    assert isinstance(out['accuracy'], float)
    assert isinstance(out['accuracy_baseline'], float)
    assert isinstance(out['n_train'], int)
    assert isinstance(out['n_test'], int)
    assert isinstance(out['example_input_rows'], list)


def test_metric_ranges_and_floor():
    out = run(42)
    acc = out['accuracy']
    acc_base = out['accuracy_baseline']
    # basic ranges
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= acc_base <= 1.0
    # Conservative performance floor for the engineered pipeline
    assert acc >= 0.70


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    assert a['accuracy'] == b['accuracy']
