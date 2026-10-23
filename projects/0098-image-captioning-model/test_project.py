import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    keys = {'digit_accuracy', 'caption_token_f1', 'num_train', 'num_test', 'model', 'decoder'}
    assert keys.issubset(set(out.keys()))
    assert 0.0 <= out['digit_accuracy'] <= 1.0
    assert 0.0 <= out['caption_token_f1'] <= 1.0
    assert isinstance(out['num_train'], int) and out['num_train'] > 0
    assert isinstance(out['num_test'], int) and out['num_test'] > 0


def test_metric_floor_and_ranges():
    out = run(42)
    # Conservative floor on caption token F1 that this pipeline comfortably passes
    assert out['caption_token_f1'] >= 0.90
    assert out['digit_accuracy'] >= 0.90


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    # determinism on main metric
    assert a['caption_token_f1'] == b['caption_token_f1']
