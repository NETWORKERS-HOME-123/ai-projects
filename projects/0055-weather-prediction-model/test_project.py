import json
from project import run


def test_run_returns_keys():
    out = run(42)
    assert isinstance(out, dict)
    keys = {"horizon_hours", "model", "n_samples", "test_size", "mae", "r2", "persistence_mae", "climatology_mae"}
    assert keys.issubset(out.keys())


def test_metrics_sane_ranges():
    out = run(42)
    # MAE should be non-negative and not absurdly large for synthetic data
    assert 0.0 <= out["mae"] < 20.0
    assert -1.0 <= out["r2"] <= 1.0
    assert 0.0 <= out["persistence_mae"] < 30.0


def test_metric_floor_r2():
    out = run(42)
    # Conservative floor: model should explain some variance on the synthetic seasonal signal
    assert out["r2"] >= 0.30


def test_determinism():
    a = run(42)
    b = run(42)
    # Compare main metric values for exact equality (deterministic pipeline)
    assert a["r2"] == b["r2"]
