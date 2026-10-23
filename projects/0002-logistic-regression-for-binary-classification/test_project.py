import json
from project import run


def test_keys():
    out = run(42)
    assert isinstance(out, dict)
    expected = {"n_train", "n_test", "threshold", "accuracy", "precision", "recall", "f1", "confusion", "threshold_comparison"}
    assert expected.issubset(set(out.keys()))


def test_metric_ranges():
    out = run(42)
    for k in ("accuracy", "precision", "recall", "f1"):
        v = out[k]
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0
    conf = out["confusion"]
    for name in ("tn", "fp", "fn", "tp"):
        assert isinstance(conf[name], int)
        assert conf[name] >= 0


def test_metric_floor_accuracy():
    out = run(42)
    # Conservative floor the pipeline should comfortably pass on this synthetic data
    assert out["accuracy"] >= 0.70


def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
