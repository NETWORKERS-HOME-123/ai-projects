import json
from project import run


def test_run_keys_and_types():
    r = run(42)
    assert isinstance(r, dict)
    expected = {
        "mae_holdout",
        "bias_holdout",
        "seasonal_naive_mae",
        "intermittent_mae",
        "stockout_mae",
        "horizon",
        "n_items",
        "forecasts_sample",
    }
    assert expected.issubset(set(r.keys()))
    assert isinstance(r["mae_holdout"], float)
    assert isinstance(r["horizon"], int)


def test_metric_ranges_and_floor():
    r = run(42)
    # basic ranges
    assert 0.0 <= r["mae_holdout"] < 50.0
    assert abs(r["bias_holdout"]) < 50.0
    # conservative metric floor for teaching exercise
    assert r["mae_holdout"] <= 3.5


def test_determinism_main_metric():
    a = run(42)["mae_holdout"]
    b = run(42)["mae_holdout"]
    assert a == b
