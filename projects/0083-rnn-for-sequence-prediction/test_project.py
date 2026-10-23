import json
from project import run
import numpy as np

def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    for k in ("r2", "mse", "baseline_mse", "sequence_replay", "model"):
        assert k in out
    assert isinstance(out["r2"], float)
    assert isinstance(out["mse"], float)
    assert isinstance(out["baseline_mse"], float)
    assert isinstance(out["sequence_replay"], str)


def test_metric_ranges():
    out = run(42)
    assert out["mse"] >= 0
    assert out["baseline_mse"] >= 0


def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    assert a["r2"] == b["r2"]  # determinism
    # conservative floor: held-out r2 should be at least 0.3
    assert a["r2"] >= 0.3
