import json

import pytest

from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    # expected keys
    keys = {"macro_f1_time", "accuracy_time", "macro_f1_publisher", "accuracy_publisher", "top_tokens_per_class", "sample_suggestions"}
    assert keys.issubset(set(res.keys()))
    assert isinstance(res["macro_f1_time"], float)
    assert isinstance(res["accuracy_time"], float)


def test_metric_ranges():
    res = run(42)
    # metrics should be between 0 and 1
    for k in ["macro_f1_time", "accuracy_time", "macro_f1_publisher", "accuracy_publisher"]:
        v = res[k]
        assert 0.0 <= v <= 1.0


def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    # determinism for the main metric
    assert a["macro_f1_time"] == b["macro_f1_time"]
    # conservative floor that this synthetic pipeline should comfortably pass
    assert a["macro_f1_time"] >= 0.6


if __name__ == "__main__":
    pytest.main([__file__])
