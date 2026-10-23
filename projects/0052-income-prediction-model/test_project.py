import json
from project import run
import math


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    # Expected keys
    keys = {"n_samples", "n_train", "n_test", "r2", "rmse", "coef_mean_abs", "rmse_gender_0", "rmse_gender_1", "analysis_note"}
    assert keys.issubset(set(out.keys()))
    # Types
    assert isinstance(out["n_samples"], int)
    assert isinstance(out["n_test"], int)
    assert isinstance(out["r2"], float)
    assert isinstance(out["rmse"], float)


def test_metric_ranges_and_floor():
    out = run(42)
    r2 = out["r2"]
    rmse = out["rmse"]
    # Reasonable ranges for this synthetic setup
    assert -1.0 <= r2 <= 1.0
    assert 0.0 <= rmse < 100.0
    # Conservative metric floor for the held-out R2 that should be comfortably met
    assert r2 >= 0.30


def test_determinism_of_main_metric():
    a = run(42)
    b = run(42)
    # Ensure determinism for the main metric and subgroup values
    assert a["r2"] == b["r2"]
    assert a["rmse"] == b["rmse"]
    assert a["rmse_gender_0"] == b["rmse_gender_0"]
    assert a["rmse_gender_1"] == b["rmse_gender_1"]
