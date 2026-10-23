import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    keys = {"num_examples", "num_train", "num_test", "accuracy", "macro_f1", "panel_inspection", "review_panel"}
    assert keys.issubset(set(out.keys()))
    assert isinstance(out["num_examples"], int)
    assert isinstance(out["accuracy"], float)
    assert isinstance(out["macro_f1"], float)


def test_metric_ranges_and_floor():
    out = run(42)
    acc = out["accuracy"]
    f1 = out["macro_f1"]
    # Ranges
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0
    # Conservative floor for this synthetic baseline
    assert f1 >= 0.65


def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
