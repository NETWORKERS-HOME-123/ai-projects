import math
from project import run


def test_run_keys_and_types():
    r = run(42)
    expected_keys = [
        "mae_horizon_1", "r2_horizon_1", "baseline_last_r2_h1", "baseline_last_mae_h1", "baseline_return_r2_h1", "baseline_return_mae_h1",
        "mae_horizon_5", "r2_horizon_5", "baseline_last_r2_h5", "baseline_last_mae_h5", "baseline_return_r2_h5", "baseline_return_mae_h5",
        "mae_horizon_10", "r2_horizon_10", "baseline_last_r2_h10", "baseline_last_mae_h10", "baseline_return_r2_h10", "baseline_return_mae_h10",
        "avg_transaction_cost_per_prediction", "note"
    ]
    for k in expected_keys:
        assert k in r
    # Types
    assert isinstance(r["mae_horizon_5"], float)
    assert isinstance(r["r2_horizon_5"], float)
    assert isinstance(r["note"], str)


def test_metric_ranges_and_floor():
    r = run(42)
    # R2 should be finite and within reasonable bounds
    for h in (1, 5, 10):
        r2 = r[f"r2_horizon_{h}"]
        assert math.isfinite(r2)
        assert r2 > -10.0 and r2 < 1.0
    # Conservative metric floor on main held-out metric r2_horizon_5
    assert r["r2_horizon_5"] >= 0.05


def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
