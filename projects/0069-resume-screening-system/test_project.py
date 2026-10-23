import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    assert "metrics" in out and "evidence_preview" in out
    metrics = out["metrics"]
    required = ["micro_f1", "micro_precision", "micro_recall", "macro_f1", "num_test_samples"]
    for k in required:
        assert k in metrics
        if k != "num_test_samples":
            assert isinstance(metrics[k], float)
        else:
            assert isinstance(metrics[k], int)


def test_metrics_range_and_floor():
    out = run(42)
    m = out["metrics"]
    # ranges
    assert 0.0 <= m["micro_f1"] <= 1.0
    assert 0.0 <= m["micro_precision"] <= 1.0
    assert 0.0 <= m["micro_recall"] <= 1.0
    assert 0.0 <= m["macro_f1"] <= 1.0
    # conservative floor: micro_f1 should be reasonable on synthetic predictable data
    assert m["micro_f1"] >= 0.70


def test_determinism_main_metric():
    a = run(42)["metrics"]["micro_f1"]
    b = run(42)["metrics"]["micro_f1"]
    assert a == b
