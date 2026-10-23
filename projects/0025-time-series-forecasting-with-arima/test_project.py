import json
from project import run
import math


def test_run_returns_expected_keys():
    out = run(42)
    assert isinstance(out, dict)
    keys = {"model", "d", "p", "mae", "naive_mae", "r2", "coverage_95", "mae_h1", "mae_h7", "mae_h14", "mae_h30", "resid_std"}
    assert keys.issubset(set(out.keys()))


def test_metric_ranges_and_floor():
    out = run(42)
    # ranges
    assert out["mae"] >= 0.0
    assert out["naive_mae"] >= 0.0
    assert 0.0 <= out["coverage_95"] <= 1.0
    # conservative floor on main held-out score r2 (allowing modest predictive power)
    assert out["r2"] >= 0.0


def test_determinism_of_main_metric():
    a = run(42)
    b = run(42)
    # determinism for r2: exact equality expected because RNG is fixed
    assert a["r2"] == b["r2"]
