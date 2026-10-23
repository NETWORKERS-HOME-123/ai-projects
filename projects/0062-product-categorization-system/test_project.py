import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    expected_keys = {"macro_f1", "unknown_rate", "sibling_confusion", "top_k_example"}
    assert expected_keys.issubset(set(out.keys()))
    assert isinstance(out["macro_f1"], float)
    assert isinstance(out["unknown_rate"], float)
    assert isinstance(out["sibling_confusion"], float)
    assert isinstance(out["top_k_example"], list)


def test_metrics_ranges_and_floor():
    out = run(42)
    # Reasonable ranges
    assert 0.0 <= out["macro_f1"] <= 1.0
    assert 0.0 <= out["unknown_rate"] <= 1.0
    assert 0.0 <= out["sibling_confusion"] <= 1.0
    # Conservative floor: expect macro F1 at least 0.70 on this synthetic dataset
    assert out["macro_f1"] >= 0.70


def test_determinism_of_main_metric():
    a = run(42)
    b = run(42)
    assert a["macro_f1"] == b["macro_f1"]
