import json
from project import run


def test_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    for k in ['n_samples', 'n_test', 'n_classes', 'word_accuracy', 'wer', 'cer', 'transcript_preview']:
        assert k in out
    assert isinstance(out['n_samples'], int)
    assert isinstance(out['word_accuracy'], float)


def test_metric_ranges_and_floor():
    out = run(42)
    assert 0.0 <= out['word_accuracy'] <= 1.0
    assert 0.0 <= out['wer'] <= 10.0
    assert 0.0 <= out['cer'] <= 10.0
    assert out['word_accuracy'] >= 0.70


def test_determinism():
    a = run(42)
    b = run(42)
    assert a['word_accuracy'] == b['word_accuracy']
