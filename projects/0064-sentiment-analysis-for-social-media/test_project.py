import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    keys = {"overall_macro_f1", "per_language_macro_f1", "per_period_macro_f1", "n_train", "n_test", "examples_uncertain"}
    assert keys.issubset(set(out.keys()))
    assert 0.0 <= out["overall_macro_f1"] <= 1.0
    assert isinstance(out["n_train"], int) and out["n_train"] > 0
    assert isinstance(out["n_test"], int) and out["n_test"] > 0


def test_metric_floor_and_ranges():
    out = run(42)
    # Conservative floor: teaching baseline should comfortably exceed 0.60 on this synthetic data
    assert out["overall_macro_f1"] >= 0.60
    # Per-language scores are between 0 and 1 or None
    for v in out["per_language_macro_f1"].values():
        if v is not None:
            assert 0.0 <= v <= 1.0
    for v in out["per_period_macro_f1"].values():
        if v is not None:
            assert 0.0 <= v <= 1.0


def test_determinism():
    a = run(42)
    b = run(42)
    # Main metric deterministic
    assert a["overall_macro_f1"] == b["overall_macro_f1"]
