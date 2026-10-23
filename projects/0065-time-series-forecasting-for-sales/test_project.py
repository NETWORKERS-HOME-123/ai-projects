import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    keys = {"model", "horizon_days", "origins_evaluated", "mae", "baseline_mae", "neg_mae", "bias", "seasonal_failure_rate", "sample_forecasts"}
    assert keys.issubset(set(out.keys()))
    assert isinstance(out["mae"], float)
    assert isinstance(out["neg_mae"], float)
    assert isinstance(out["origins_evaluated"], int)


def test_metric_ranges_and_floor():
    out = run(42)
    # basic ranges
    assert out["mae"] >= 0.0
    assert 0.0 <= out["seasonal_failure_rate"] <= 1.0
    # conservative performance floor on main metric (neg_mae): expect neg_mae >= -150
    assert out["neg_mae"] >= -150.0


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    assert a["neg_mae"] == b["neg_mae"]
