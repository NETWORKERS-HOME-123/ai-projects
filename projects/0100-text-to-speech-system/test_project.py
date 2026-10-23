import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    for k in ("r2_train", "r2_test", "mae_test", "example_wave_mean"):
        assert k in out
        assert isinstance(out[k], float)


def test_metric_ranges_and_floor():
    out = run(42)
    # r2 values are usually between -inf and 1. We expect a modest positive performance.
    assert -1.0 <= out["r2_train"] <= 1.0
    assert -1.0 <= out["r2_test"] <= 1.0
    assert out["mae_test"] >= 0.0
    # conservative floor for held-out R2 that should be comfortably met by this synthetic pipeline
    assert out["r2_test"] >= 0.30


def test_determinism_of_main_metric():
    a = run(42)
    b = run(42)
    # check determinism for the main metric
    assert a["r2_test"] == b["r2_test"]
