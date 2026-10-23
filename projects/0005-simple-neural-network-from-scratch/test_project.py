import json
from project import run

def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    keys = {"train_loss", "test_loss", "test_accuracy", "baseline_accuracy", "grad_check_diff"}
    assert set(out.keys()) == keys
    for k in keys:
        assert isinstance(out[k], float)

def test_metric_ranges_and_floor():
    out = run(42)
    # Losses should be finite and non-negative
    assert out["train_loss"] >= 0.0 and out["train_loss"] < 5.0
    assert out["test_loss"] >= 0.0 and out["test_loss"] < 5.0
    # Accuracy ranges
    assert 0.0 <= out["test_accuracy"] <= 1.0
    assert 0.0 <= out["baseline_accuracy"] <= 1.0
    # Conservative floor for held-out accuracy (should comfortably pass on this toy task)
    assert out["test_accuracy"] >= 0.75
    # Grad check difference should be small
    assert out["grad_check_diff"] < 1e-2

def test_determinism():
    a = run(42)
    b = run(42)
    # Compare main metric
    assert a["test_accuracy"] == b["test_accuracy"]
