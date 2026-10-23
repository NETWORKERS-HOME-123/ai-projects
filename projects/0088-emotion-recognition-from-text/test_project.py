import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    # expected keys
    keys = {"macro_f1", "per_label_f1", "n_train", "n_test", "classification_report", "ambiguous_examples_for_review"}
    assert keys.issubset(set(res.keys()))
    assert isinstance(res["macro_f1"], float)
    assert isinstance(res["per_label_f1"], dict)
    assert isinstance(res["n_train"], int)
    assert isinstance(res["n_test"], int)
    assert isinstance(res["classification_report"], str)
    assert isinstance(res["ambiguous_examples_for_review"], list)


def test_metric_ranges_and_floor():
    res = run(42)
    # Macro F1 should be a sensible probability between 0 and 1
    assert 0.0 <= res["macro_f1"] <= 1.0
    # Per-label F1 values should be in range
    for v in res["per_label_f1"].values():
        assert 0.0 <= v <= 1.0
    # Conservative floor: expect macro_f1 at least 0.55 for this synthetic task
    assert res["macro_f1"] >= 0.55


def test_determinism_of_main_metric():
    a = run(42)["macro_f1"]
    b = run(42)["macro_f1"]
    assert a == b
