import json
from project import run

def test_run_returns_keys():
    r = run(42)
    assert isinstance(r, dict)
    keys = {"val_loss", "val_accuracy", "sample_temperature_0.5", "sample_temperature_1.0", "sample_temperature_2.0", "vocab_size", "epochs"}
    assert keys.issubset(set(r.keys()))

def test_metrics_ranges():
    r = run(42)
    assert 0.0 <= r["val_loss"] < 100.0
    assert 0.0 <= r["val_accuracy"] <= 1.0
    assert isinstance(r["sample_temperature_1.0"], str)
    assert r["vocab_size"] >= 1

def test_metric_floor_and_determinism():
    a1 = run(42)["val_accuracy"]
    a2 = run(42)["val_accuracy"]
    assert a1 == a2
    assert a1 >= 0.15
