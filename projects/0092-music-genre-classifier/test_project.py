import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    for k in ("macro_f1", "accuracy", "n_train", "n_test", "model"):
        assert k in res
    assert isinstance(res["macro_f1"], float)
    assert isinstance(res["accuracy"], float)


def test_metric_ranges_and_floor():
    res = run(42)
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert 0.0 <= res["accuracy"] <= 1.0
    # Conservative performance floor for this synthetic task
    assert res["macro_f1"] >= 0.75


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    # main metric determinism
    assert a["macro_f1"] == b["macro_f1"]
