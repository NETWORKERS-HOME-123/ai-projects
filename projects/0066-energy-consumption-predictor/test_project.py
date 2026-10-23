import json
from project import run
import math


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    keys = {"r2", "mae", "baseline_mae", "peak_mae", "holiday_mae", "missing_meter_warning", "stale_input_warning", "load_planning_chart"}
    assert keys.issubset(set(res.keys()))
    assert isinstance(res["r2"], float)
    assert isinstance(res["mae"], float)
    assert isinstance(res["baseline_mae"], float)
    assert isinstance(res["missing_meter_warning"], int)
    assert isinstance(res["stale_input_warning"], int)
    assert isinstance(res["load_planning_chart"], dict)


def test_metric_ranges_and_floor():
    res = run(42)
    # r2 should be a finite float and reasonably positive for this synthetic baseline
    r2 = res["r2"]
    assert math.isfinite(r2)
    assert -1.0 <= r2 <= 1.0
    # Conservative main metric floor: r2 >= 0.30
    assert r2 >= 0.30
    # MAE positive and baseline_mae should be positive
    assert res["mae"] >= 0.0
    assert res["baseline_mae"] >= 0.0


def test_determinism_of_main_metric():
    a = run(42)["r2"]
    b = run(42)["r2"]
    assert a == b
