import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    expected_keys = {"best_val_loss", "best_params", "best_val_loss_random", "trials", "budget"}
    assert expected_keys.issubset(set(res.keys()))
    assert isinstance(res["best_val_loss"], float)
    assert isinstance(res["best_params"], dict)
    assert isinstance(res["trials"], list)


def test_metric_ranges_and_floor():
    res = run(42)
    b = res["best_val_loss"]
    br = res["best_val_loss_random"]
    # losses must be non-negative and finite
    assert b >= 0.0 and b < float("inf")
    assert br >= 0.0 and br < float("inf")
    # conservative floor: best_val_loss should be reasonably low for this synthetic task
    assert b <= 0.9


def test_determinism_of_main_metric():
    a = run(42)["best_val_loss"]
    b = run(42)["best_val_loss"]
    assert a == b
